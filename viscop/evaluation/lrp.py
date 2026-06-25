import argparse
import os
import random
import traceback
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
import imageio

sys.path.append(".")
from viscop import disable_torch_init
from evaluation.benchmarks import build_dataset
from evaluation.register import INFERENCES
from evaluation.utils import CUDADataLoader

# ==========================================
# 1. Relevance Utils
# ==========================================

def make_attention_relevance(attn, d_attn, eps=1e-6):
    return F.relu(d_attn) * attn

def overlay_positive_heatmap_on_image(img_pil, pos_map, alpha=0.5):
    m = np.clip(pos_map, 0, 1)
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", [(0, "white"), (1, "red")])
    heat_rgb = (cmap(norm(m))[..., :3] * 255).astype(np.uint8)
    img = np.array(img_pil.resize((m.shape[1], m.shape[0]), Image.BICUBIC)).astype(np.float32)
    out = (alpha * heat_rgb + (1 - alpha) * img).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)

@torch.no_grad()
def tokens_to_heatmap(cls_to_tokens, grid_size_hw, out_size=(224, 224)):
    Htok, Wtok = grid_size_hw
    if cls_to_tokens.shape[0] == (Htok * Wtok) + 1:
        pr = cls_to_tokens[1:]
    else:
        pr = cls_to_tokens
    
    pr = pr.reshape(Htok, Wtok).float()
    pr = pr - pr.min()
    if pr.max() > 0:
        pr = pr / pr.max()
    pr = pr[None, None, :, :]
    pr = F.interpolate(pr, size=out_size, mode='bilinear', align_corners=False)[0, 0]
    return pr.clamp(0, 1).cpu().numpy()

# ==========================================
# 2. Optimized Hook Class
# ==========================================

class SigLIPWithAttnHooks(nn.Module):
    def __init__(self, vision_model: nn.Module, mode='all', device='cuda'):
        super().__init__()
        self.model = vision_model
        self.device = device
        self.mode = mode 
        self.attn_vars = []        
        self.interaction_vars = [] 
        self._monkey_patch()

    def _monkey_patch(self):
        enc_blocks = self._find_blocks(['vision_encoder', 'encoder', 'layers'], ['encoder', 'layers'], ['layers'])
        int_blocks = self._find_blocks(['vision_encoder', 'encoder', 'interaction_modules'], ['encoder', 'interaction_modules'], ['interaction_modules'])

        if (self.mode == 'encoder' or self.mode == 'all') and enc_blocks:
            print(f"   Hooking {len(enc_blocks)} Encoder layers...")
            for blk in enc_blocks:
                self._hook_self_attn(blk)
        
        if (self.mode == 'interaction' or self.mode == 'all') and int_blocks:
            print(f"   Hooking {len(int_blocks)} Interaction layers...")
            for blk in int_blocks:
                self._hook_interaction_attn(blk)

    def _find_blocks(self, *candidate_paths):
        for path in candidate_paths:
            curr = self.model
            valid = True
            for attr in path:
                if hasattr(curr, attr): curr = getattr(curr, attr)
                else: valid = False; break
            if valid and isinstance(curr, nn.ModuleList):
                return curr
        return None

    def _hook_self_attn(self, blk):
        attn_mod = getattr(blk, 'self_attn', None)
        if attn_mod is None: return
        
        orig_forward = attn_mod.forward

        def wrapped_forward(x, *args, **kwargs):
            if isinstance(x, torch.Tensor): hidden_states = x
            else: hidden_states = args[0] if args else kwargs.get('hidden_states')

            cu_seqlens = kwargs.get('cu_seqlens', None)
            
            q_weight = attn_mod.q_proj.weight
            k_weight = attn_mod.k_proj.weight
            v_weight = attn_mod.v_proj.weight
            q_bias = attn_mod.q_proj.bias
            k_bias = attn_mod.k_proj.bias
            v_bias = attn_mod.v_proj.bias

            def compute_attn(h_states):
                B, N, C = h_states.shape
                if hasattr(attn_mod, 'num_heads'):
                    num_heads = attn_mod.num_heads
                    head_dim = getattr(attn_mod, 'head_dim', C // num_heads)
                else:
                    num_heads = C // 64; head_dim = 64

                q = F.linear(h_states, q_weight, q_bias)
                k = F.linear(h_states, k_weight, k_bias)
                v = F.linear(h_states, v_weight, v_bias)

                q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
                k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
                v = v.view(B, N, num_heads, head_dim).transpose(1, 2)

                attn_weights = torch.matmul(q, k.transpose(-2, -1))
                attn_weights = attn_weights * (head_dim ** -0.5)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

                if torch.is_grad_enabled():
                    attn_weights.retain_grad()
                    self.attn_vars.append(attn_weights)

                out = torch.matmul(attn_weights, v)
                out = out.transpose(1, 2).contiguous().reshape(B, N, C)
                return out

            if hidden_states.dim() == 2 and cu_seqlens is not None:
                outputs = []
                for i in range(len(cu_seqlens) - 1):
                    start, end = cu_seqlens[i], cu_seqlens[i+1]
                    chunk = hidden_states[start:end].unsqueeze(0) 
                    out_chunk = compute_attn(chunk) 
                    outputs.append(out_chunk.squeeze(0))
                attn_output = torch.cat(outputs, dim=0)
            elif hidden_states.dim() == 2:
                attn_output = compute_attn(hidden_states.unsqueeze(0)).squeeze(0)
            else:
                attn_output = compute_attn(hidden_states)

            if hasattr(attn_mod, 'out_proj'):
                attn_output = attn_mod.out_proj(attn_output)
            
            return attn_output

        attn_mod.forward = wrapped_forward

    def _hook_interaction_attn(self, blk):
        attn_mod = blk 
        orig_forward = attn_mod.forward

        def wrapped_interaction_forward(visual_probes, hidden_states, *args, **kwargs):
            vp = visual_probes
            if vp.dim() == 2: vp = vp.unsqueeze(0)
            hs = hidden_states
            if hs.dim() == 2: hs = hs.unsqueeze(0)

            B, N_q, _ = vp.shape
            B, N_k, _ = hs.shape
            num_heads = attn_mod.num_heads
            head_dim = attn_mod.head_dim

            q = attn_mod.q_proj(vp)
            k = attn_mod.k_proj(hs)
            v = attn_mod.v_proj(hs)

            q = q.view(B, N_q, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, N_k, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N_k, num_heads, head_dim).transpose(1, 2)

            attn_weights = torch.matmul(q, k.transpose(-2, -1)) 
            attn_weights = attn_weights * (head_dim ** -0.5)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if torch.is_grad_enabled():
                attn_weights.retain_grad()
                self.interaction_vars.append(attn_weights)

            attn_output = torch.matmul(attn_weights, v) 
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, N_q, -1)
            
            if hasattr(attn_mod, 'out_proj'):
                attn_output = attn_mod.out_proj(attn_output)
            
            if visual_probes.dim() == 2:
                attn_output = attn_output.squeeze(0)
            
            return attn_output

        attn_mod.forward = wrapped_interaction_forward

    def clear(self):
        self.attn_vars = []
        self.interaction_vars = []

# ==========================================
# 3. Main Logic
# ==========================================

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    disable_torch_init()
    
    model_init, mm_infer = INFERENCES(args.model_path)
    device = "cuda"
    model, processor = model_init(args.model_path, args.max_visual_tokens, device_map=device)
    model.eval()

    vision_tower_inner = None
    if hasattr(model, 'model') and hasattr(model.model, 'vision_encoder'):
        vision_tower_inner = model.model.vision_encoder.vision_encoder
    
    if vision_tower_inner is None:
        print("Error locating vision tower."); return

    dataset = build_dataset(args.benchmark, data_root=args.data_root, processor=processor, num_splits=1, split_idx=0, fps=args.fps, max_frames=args.max_frames)
    dataloader = CUDADataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True, collate_fn=lambda x: x[0], pin_memory=True)
    modal = dataset.MODAL

    print(f"Processing {len(dataloader)} samples...")
    for idx, data in enumerate(tqdm(dataloader)):
        if idx  > 0: break
        data_ids = data["data_ids"]
        text_inputs = data["text_inputs"]
        image_inputs = data["image_inputs"]
        
        for data_id, text_input in zip(data_ids, text_inputs):
            try:
                print(f"\nProcessing ID: {data_id}")
                cleanup()
                data_dict = {**image_inputs, **text_input}
                response = mm_infer(data_dict, model=model, tokenizer=processor.tokenizer, modal=modal, do_sample=False)
                print(f"Model Response: {response}")

                # === ROBUST INPUT PREPARATION ===
                # Get correct dtype
                model_dtype = model.model.embed_tokens.weight.dtype if hasattr(model, 'model') else torch.bfloat16
                
                input_kwargs = {**image_inputs, **text_input}
                input_kwargs['modals'] = [modal]
                
                # Iterate over ALL items to ensure EVERYTHING is on CUDA
                for k, v in input_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'pixel_values':
                            input_kwargs[k] = v.to(device=device, dtype=model_dtype)
                        else:
                            input_kwargs[k] = v.to(device=device)
                    elif isinstance(v, list):
                        # Handle lists of tensors (e.g., grid_sizes often come as lists of Tensors)
                        if len(v) > 0 and isinstance(v[0], torch.Tensor):
                             input_kwargs[k] = [t.to(device=device) for t in v]

                # Get Target
                with torch.no_grad():
                    outputs = model(**input_kwargs)
                    logits = outputs.logits
                    pred_token_id = logits[0, args.target_token_idx].argmax()

                relevance_map = None

                # Step 2: Interaction LRP
                print("   Running Interaction LRP...")
                cleanup()
                wrapper_int = SigLIPWithAttnHooks(vision_tower_inner, mode='interaction', device=device)
                model.zero_grad()
                for param in wrapper_int.model.parameters(): param.requires_grad = True
                
                outputs = model(**input_kwargs)
                score = outputs.logits[0, args.target_token_idx, pred_token_id]
                score.backward()
                
                inter_attns = [a.detach() for a in wrapper_int.interaction_vars]
                inter_grads = [g.detach() for g in wrapper_int.interaction_vars]
                
                if len(inter_attns) > 0:
                    total_R = 0
                    for A, dA in zip(inter_attns, inter_grads):
                        R = make_attention_relevance(A, dA) 
                        R = R.sum(dim=1).sum(dim=1) 
                        total_R = total_R + R
                    relevance_map = total_R
                    print("   Got Interaction Relevance.")
                
                wrapper_int.clear()
                del wrapper_int
                cleanup()

                # Step 3: Encoder LRP (Fallback)
                if relevance_map is None:
                    print("   Interaction empty. Running Encoder LRP...")
                    wrapper_enc = SigLIPWithAttnHooks(vision_tower_inner, mode='encoder', device=device)
                    model.zero_grad()
                    for param in wrapper_enc.model.parameters(): param.requires_grad = True
                    
                    outputs = model(**input_kwargs)
                    score = outputs.logits[0, args.target_token_idx, pred_token_id]
                    score.backward()
                    
                    attn_list = [a.detach() for a in wrapper_enc.attn_vars]
                    grad_list = [g.detach() for g in wrapper_enc.attn_vars]
                    
                    if len(attn_list) > 0:
                        R_last = make_attention_relevance(attn_list[-1], grad_list[-1])
                        relevance_map = R_last.sum(dim=1).mean(dim=1) 
                        print("   Got Encoder Relevance.")

                    wrapper_enc.clear()
                    del wrapper_enc
                    cleanup()

# ---------------------------------------------------
                # VISUALIZATION (Fixed for Flattened Inputs)
                # ---------------------------------------------------
                if relevance_map is not None:
                    relevance_map = relevance_map.float().cpu()
                    
                    # Defaults
                    grid_h, grid_w = 16, 16 # Default 224/14
                    
                    # 1. Try to Determine Grid Dimensions from Input
                    if 'pixel_values' in input_kwargs:
                        pv = input_kwargs['pixel_values']
                        if isinstance(pv, list): pv = pv[0]
                        
                        # --- FIX: Handle 2D Flattened Inputs [N, C] ---
                        if pv.dim() == 2:
                            # Cannot infer H/W from [N, C] reliably without metadata.
                            # We assume standard 336px or 224px based on token count
                            # or just use the relevance map to decide.
                            pass 
                        elif pv.dim() == 5:
                            # Standard Video: [B, C, T, H, W]
                            if pv.shape[1] == 3: _, _, _, H_in, W_in = pv.shape
                            else: _, _, _, H_in, W_in = pv.shape
                            grid_h, grid_w = H_in // 14, W_in // 14
                        elif pv.dim() == 4:
                            # Image: [B, 3, H, W]
                            _, _, H_in, W_in = pv.shape
                            grid_h, grid_w = H_in // 14, W_in // 14

                    # 2. Refine Grid based on Relevance Map Size
                    # Relevance Map is [T, N_tokens] or [Total_Tokens]
                    total_tokens = relevance_map.numel()
                    
                    # Heuristic: If we have a 1D map, try to make it square per frame
                    if relevance_map.dim() == 1:
                        # Try to find T such that N_tokens is a square number
                        # Common grid sizes: 14x14=196, 16x16=256, 24x24=576
                        common_grids = [14, 16, 24, 28, 32]
                        found = False
                        for g in common_grids:
                            sq = g*g
                            if total_tokens % sq == 0:
                                T_est = total_tokens // sq
                                # Sanity check: T should be reasonable (e.g. < 1000)
                                if T_est < 1000:
                                    grid_h, grid_w = g, g
                                    relevance_map = relevance_map.view(T_est, sq)
                                    found = True
                                    print(f"   [Grid Search] Inferred {T_est} frames of {g}x{g} tokens.")
                                    break
                        
                        if not found:
                            # Fallback to single square
                            side = int(np.sqrt(total_tokens))
                            relevance_map = relevance_map.view(1, total_tokens)
                            grid_h, grid_w = side, side
                    
                    # If map is already [T, N], check if N matches grid
                    elif relevance_map.dim() == 2:
                        T_curr, N_curr = relevance_map.shape
                        side = int(np.sqrt(N_curr))
                        if side * side == N_curr:
                            grid_h, grid_w = side, side
                        elif (side * side) != N_curr:
                             # Try side-1 (sometimes CLS token is missing)
                             side_s = int(np.sqrt(N_curr + 1))
                             if side_s * side_s == (N_curr + 1):
                                 grid_h, grid_w = side_s, side_s

                    # 3. Resolve Video Path (Using Dataset Lookup)
                    raw_path = None
                    # A. Check Batch
                    if 'video_path' in data: raw_path = data['video_path']
                    elif 'image_path' in data: raw_path = data['image_path']
                    # B. Check Dataset Dict (Robust)
                    if raw_path is None and hasattr(dataset, 'data_dict') and data_id in dataset.data_dict:
                        entry = dataset.data_dict[data_id]
                        raw_path = entry.get('video_path', entry.get('image_path'))
                    
                    if isinstance(raw_path, list): raw_path = raw_path[0]

                    # C. Check Filesystem
                    full_video_path = None
                    if raw_path:
                        if os.path.exists(raw_path): full_video_path = raw_path
                        elif os.path.exists(os.path.join(args.data_root, raw_path)):
                            full_video_path = os.path.join(args.data_root, raw_path)
                    
                    if full_video_path is None:
                        print(f"   [Warning] Could not find video file. Raw: {raw_path}")

                    # 4. Generate & Save
                    if full_video_path and full_video_path.endswith(('.mp4', '.avi', '.mkv', '.webm')):
                        try:
                            vr = VideoReader(full_video_path, ctx=cpu(0))
                            T_map = relevance_map.shape[0]
                            
                            # Map LRP frames to Video frames
                            frame_indices = np.linspace(0, len(vr)-1, T_map).astype(int)
                            
                            video_frames_out = []
                            original_frames_out = []
                            
                            for t in range(T_map):
                                heatmap = tokens_to_heatmap(relevance_map[t], (grid_h, grid_w))
                                img_np = vr[frame_indices[t]].asnumpy()
                                img_pil = Image.fromarray(img_np)
                                
                                original_frames_out.append(img_np)
                                vis = overlay_positive_heatmap_on_image(img_pil, heatmap, alpha=0.6)
                                video_frames_out.append(np.array(vis))
                            
                            clean_id = str(data_id).replace("/", "_")
                            fname_lrp = f"{args.benchmark}_{clean_id}_lrp.mp4"
                            fname_orig = f"{args.benchmark}_{clean_id}_orig.mp4"
                            
                            imageio.mimsave(os.path.join(args.save_path, fname_lrp), video_frames_out, fps=args.fps, format='FFMPEG')
                            imageio.mimsave(os.path.join(args.save_path, fname_orig), original_frames_out, fps=args.fps, format='FFMPEG')
                            print(f"   Saved LRP: {fname_lrp}")
                            print(f"   Saved Orig: {fname_orig}")

                        except Exception as e:
                            print(f"   [Video Generation Error] {e}")
                            traceback.print_exc()

                    else:
                        # Fallback Image
                        best_frame_idx = relevance_map.sum(dim=1).argmax().item()
                        heatmap = tokens_to_heatmap(relevance_map[best_frame_idx], (grid_h, grid_w))
                        
                        img_pil = Image.open(full_video_path).convert("RGB") if full_video_path else Image.new("RGB", (224, 224), (200,200,200))
                        
                        vis = overlay_positive_heatmap_on_image(img_pil, heatmap, alpha=0.6)
                        
                        clean_id = str(data_id).replace("/", "_")
                        fname = f"{args.benchmark}_{clean_id}_vis.jpg"
                        vis.save(os.path.join(args.save_path, fname))
                        print(f"   Saved image: {fname}")

            except Exception:
                traceback.print_exc()
            finally:
                cleanup()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./vis_output")
    parser.add_argument("--target_token_idx", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=180)
    parser.add_argument("--max_visual_tokens", type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    main()