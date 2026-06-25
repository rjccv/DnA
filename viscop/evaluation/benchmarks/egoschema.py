import json
import os
import re
import time
import random
import string
import requests
from copy import deepcopy
from typing import Any, Dict, List, Union

from .base import BaseVideoEvalDataset, filter_metadata


class EgoSchemaDataset(BaseVideoEvalDataset):

    def load_data(self, data_root: str) -> Dict[int, Any]:
        data_dict = {}

        video_folder = os.path.join(data_root, "videos")
        json_file = os.path.join(data_root, "questions.json")
        with open(json_file, "r") as f:
            data_list = json.load(f)

        for data in data_list:
            # question_id = data["q_uid"]
            question_id = data["question_idx"]
            video_id = data["video_idx"]
            # for video_format in ["mp4", "avi", "mov", "mkv"]:
            #     video_path = os.path.join(video_folder, f"{question_id}.{video_format}")
            #     if os.path.exists(video_path):
            #         break
            video_path = os.path.join(video_folder, f"{video_id}.mp4")
            assert os.path.exists(video_path), f"Cannot find the video file: {video_id}"

            data_dict[question_id] = {
                # required fields for data loading
                "video_path": video_path,
                "start_time": None,
                "end_time": None,
                # custom fields for instruction generation and post processing
                "question": data["question"],
                "options": [data["options"][i] for i in range(5)],
            }

        return data_dict

    def generate_instruction(self, data_id: Union[int, str], video: Any) -> str:
        meta_data = self.data_dict[data_id]
        question = meta_data["question"]
        options = meta_data["options"]
        instruction = f'Select the best answer to the following multiple-choice question based on the video.\n{question}\nOptions:\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}\n(E) {options[4]}\nAnswer with the option\'s letter from the given choices directly and only give the best option. The best answer is: ' 
        return instruction

    def process_response(self, data_id: Union[int, str], response: str) -> int:
        options = self.data_dict[data_id]["options"]
        letters = ['A', 'B', 'C', 'D', 'E']

        response = response.replace('answer', '')
        response = response.replace('Answer', '')
        pred_answer = re.findall('[\(\ ]*[A-E][\)\ ]*', response)

        find_flag = False
        if len(pred_answer) == 0:
            for idx, opt in enumerate(options):
                opt = opt.strip()
                opt = opt.strip('.')
                if opt.lower() in response.lower():
                    pred_idx = idx
                    find_flag = True
                    break
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip('()')
            pred_idx = letters.index(pred_answer)
            find_flag = True

        assert find_flag, f"Cannot find the answer in the options: {response}"
        return pred_idx

    # def evaluate(self, results: List[Dict[str, Any]]) -> (None, Dict[str, List[Dict[str, Any]]]):
    #     url = "https://validation-server.onrender.com/api/upload/"
    #     headers = {"Content-Type": "application/json"}
    #     submission = {result["data_id"]: result["prediction"] for result in results}

    #     response = requests.post(url, headers=headers, json=submission)
    #     assert response.status_code == 200, f"Failed to send POST request. Status code: {response.status_code}"
    #     matches = re.findall(r'(\d+) correct, (\d+) wrong', response.text)
    #     assert len(matches) == 2, f"Failed to parse the response: {response.text}"
    #     print(response.text)

    #     total_correct, total_wrong = matches[0]
    #     total_correct, total_wrong = int(total_correct), int(total_wrong)
    #     subset_correct, subset_wrong = matches[1]
    #     subset_correct, subset_wrong = int(subset_correct), int(subset_wrong)
    #     metrics = {
    #         "Subset": subset_correct / (subset_correct + subset_wrong) * 100,
    #         "Total": total_correct / (total_correct + total_wrong) * 100,
    #     }

    #     infos = []
    #     for data in results:
    #         data = deepcopy(data)
    #         meta_data = deepcopy(self.data_dict[data["data_id"]])
    #         infos.append(
    #             {
    #                 **data,
    #                 "meta_data": filter_metadata(meta_data),
    #             }
    #         )
    #     infos = [metrics] + infos

    #     return metrics, infos
    

    def evaluate(self, results: List[Dict[str, Any]], model_name: str) -> (Dict[str, float], List[Dict[str, Any]]):
        url = "https://validation-server.onrender.com/api/upload/"
        headers = {"Content-Type": "application/json"}
        submission = {result["data_id"]: result["prediction"] for result in results}

        try:
            response = requests.post(url, headers=headers, json=submission, timeout=30)

            if response.status_code != 200:
                raise RuntimeError(
                    f"POST failed with status {response.status_code}: {response.text}"
                )

            matches = re.findall(r'(\d+) correct, (\d+) wrong', response.text)
            assert len(matches) == 2, f"Failed to parse the response: {response.text}"

            print(response.text)

            total_correct, total_wrong = map(int, matches[0])
            subset_correct, subset_wrong = map(int, matches[1])

            metrics = {
                "Subset": subset_correct / (subset_correct + subset_wrong) * 100,
                "Total": total_correct / (total_correct + total_wrong) * 100,
            }

        except Exception as e:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.getcwd(), "offline_submissions")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{model_name}_egoschema_submission_{timestamp}.json")
            with open(save_path, "w") as f:
                json.dump(submission, f, indent=2)

            print("\n[WARNING] Online evaluation failed.")
            print(f"Reason: {e}")
            print(f"Predictions saved locally to:\n  {save_path}")
            print("You can upload this JSON manually from a machine with internet access.\n")

            metrics = None  

        infos = []
        for data in results:
            data = deepcopy(data)
            meta_data = deepcopy(self.data_dict[data["data_id"]])
            infos.append(
                {
                    **data,
                    "meta_data": filter_metadata(meta_data),
                }
            )

        if metrics is not None:
            infos = [metrics] + infos

        return metrics, infos
