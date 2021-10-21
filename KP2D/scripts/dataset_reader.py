import os
from pathlib import Path
import cv2
import numpy as np
from shutil import copyfile
import time
import json
from joblib import Parallel, delayed
from kp2d_wrapper import Kp2dWrapper


class EvaluationMethodExecutor:

    def __init__(self,
                 method_name,
                 method_params,
                 eval_method_path,
                 folder_name,
                 output_path,
                 imgs_extension,
                 number_of_images,
                 create_wrapper_method,
                 is_region_rep=False):
        self.method_name = method_name
        self.method_params = method_params
        self.eval_method_path = eval_method_path
        self.folder_name = folder_name
        self.output_path = output_path
        self.imgs_extension = imgs_extension
        self.number_of_images = number_of_images
        self.create_wrapper_method = create_wrapper_method
        self.is_region_rep = is_region_rep

        self.keypoints = []
        self.descriptors = []
        self.imgs_sizes = []
        self.times = []

    def __get_results_dictionary(self, results, homographies, points, imgs_sizes, times):
        return {
            "folder_name": self.folder_name,
            "model_name": self.method_name,
            "model_params": self.method_params,
            "results": results,
            "homographies": homographies,
            "points": points,
            "imgs_sizes": imgs_sizes,
            "times": times
        }

    def __create_result_dictionary(self, img_id, keypoints, descriptors):
        descriptors = [[float(d) for d in desc] for desc in descriptors]
        sizes = [float(kp.size) for kp in keypoints]
        keypoints = [[float(kp.pt[0]), float(kp.pt[1])] for kp in keypoints]
        return {
            "image_id": img_id,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "sizes": sizes,
        }

    def __create_img_size_dictionary(self, img_id, width, height):
        return {
            "image_id": img_id,
            "width": width,
            "height": height
        }

    def __create_imgs_sizes_array(self):
        imgs_sizes_array = []
        for i in range(len(self.imgs_sizes)):
            imgs_sizes_array.append(self.__create_img_size_dictionary(
                i + 1, self.imgs_sizes[i][0], self.imgs_sizes[i][1]))
        return imgs_sizes_array

    def __create_homographies_array(self, homographies):
        homographies_array = []
        for i in range(len(homographies)):
            homographies_array.append({
                "from": 1,
                "to": i + 2,
                "homography": homographies[i]
            })
        return homographies_array

    def __create_points_array(self, points1, points2):
        points_dictionary = []
        points_dictionary.append({
            "img_id": 1,
            "points": [[float(p1), float(p2)] for p1, p2 in points1]
        })
        points_dictionary.append({
            "img_id": 2,
            "points": [[float(p1), float(p2)] for p1, p2 in points2]
        })
        return points_dictionary

    def __create_times_array(self):
        times = []
        for i in range(len(self.times)):
            times.append({
                "image_id": i + 1,
                "computation_time": self.times[i],
                "measurement_unit": "seconds"
            })
        return times

    def __read_homographies(self):
        def create_homography_path(n):
            return str(Path(self.eval_method_path, f"H_1_{n}"))
        homographies = []
        for i in range(self.number_of_images - 1):
            homography = []
            with open(create_homography_path(i + 2), "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    homography.append([float(l) for l in line])
            homographies.append(homography)

        return homographies

    def __read_points(self):
        def create_points_path(n):
            return str(Path(self.eval_method_path, f"points_{n}"))
        points_1 = []
        points_2 = []
        with open(create_points_path(1), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                points_1.append([float(l) for l in line])
        with open(create_points_path(2), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(" ")
                points_2.append([float(l) for l in line])
        return points_1, points_2

    def save_results(self):
        def create_metadata_path(parent_path):
            return str(Path(parent_path, "metadata.json"))

        output_directory = Path(self.output_path, self.folder_name)
        output_directory.mkdir(parents=True, exist_ok=False)
        results = []
        for i in range(len(self.keypoints)):
            results.append(self.__create_result_dictionary(
                i + 1,
                self.keypoints[i],
                self.descriptors[i]
            ))

        homographies = self.__create_homographies_array(
            self.__read_homographies())
        points = []
        if self.is_region_rep:
            points1, points2 = self.__read_points()
            points = self.__create_points_array(points1, points2)

        src_metadata_path = create_metadata_path(self.eval_method_path)
        dst_metadata_path = create_metadata_path(output_directory)
        copyfile(src_metadata_path, dst_metadata_path)

        json_to_write = self.__get_results_dictionary(
            results, homographies, points, self.__create_imgs_sizes_array(), self.__create_times_array())

        with open(str(Path(output_directory, "method_output.json")), "w") as f:
            json.dump(json_to_write, f)

    def execute(self):
        def create_img_path(n):
            return str(Path(self.eval_method_path, f"{n}.{self.imgs_extension}"))

        wrapper = self.create_wrapper_method()
        for n in range(1, self.number_of_images + 1):
            img_path = create_img_path(n)
            t = time.time()
            keypoints, descriptors, size = wrapper.process_image(img_path)
            t = time.time() - t
            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)
            self.imgs_sizes.append(size)
            self.times.append(t)


def parallel_function(method_name, method_params, folder_path, folder_name, output_directory, number_of_images, imgs_extension):
    prefix = folder_name[0]
    assert prefix in ["i", "m", "b", "g",
                      "r", "v"], f"The prefix is not valid: {folder_path}"
    is_region_rep = prefix == "r"
    number_of_images = 2 if prefix == "r" else number_of_images

    evaluation_method_executor = EvaluationMethodExecutor(
        method_name=method_name,
        method_params=method_params,
        eval_method_path=folder_path,
        folder_name=folder_name,
        output_path=output_directory,
        imgs_extension=imgs_extension,
        number_of_images=number_of_images,
        create_wrapper_method=create_wrapper_fn(method_params["conf_threshold"], method_params["model_path"]),
        is_region_rep=is_region_rep
    )
    evaluation_method_executor.execute()
    evaluation_method_executor.save_results()


def evaluate_dataset(method_name, method_params, full_paths_and_filenames, output_directory, number_of_images, imgs_extension, n_jobs=1):

    Parallel(n_jobs=n_jobs)(delayed(parallel_function)(
        method_name,
        method_params,
        folder_path,
        folder_name,
        output_directory,
        number_of_images,
        imgs_extension) for folder_path, folder_name in full_paths_and_filenames)


already_created_wrappers = {}
def create_wrapper_fn(conf_threshold, model_path):
    if str(conf_threshold) not in already_created_wrappers:
        already_created_wrappers[str(conf_threshold)] = Kp2dWrapper(
            model_path=model_path,
            conf_threshold=conf_threshold,
            k_best=1000
        )
    def f():
        return already_created_wrappers[str(conf_threshold)]
    return f


def main_method_executor(imgs_path, output_directory, method_name, method_params, imgs_extension, number_of_images, n_jobs=1):
    data_paths = os.listdir(imgs_path)
    full_paths_and_filenames = [(str(Path(imgs_path, i)), i)
                                for i in data_paths]
    t = time.time()
    evaluate_dataset(method_name, method_params,
                     full_paths_and_filenames, output_directory, number_of_images, imgs_extension, n_jobs)
    t = time.time() - t

    return t

def execute_grid(grid_values, dataset_path, output_directory, method_name, imgs_extension, number_of_images, n_jobs):
    keys = list(grid_values.keys())
    for key in keys:
        try:
            out_path = Path(output_directory, key)
            out_path.mkdir(parents=True, exist_ok=True)
            method_params = grid_values[key]["params"]
            main_method_executor(
                dataset_path,
                str(out_path),
                method_name,
                method_params,
                imgs_extension,
                number_of_images,
                n_jobs
            )
        except Exception as e:
            print(e)

if __name__ == "__main__":
    
    GRID = {
        
        "kp2d_1": {
            
            "params": {
                "model_path": "/mnt/g27prist/TCO/TCO-Studenten/ariel_research_project/pretrained_models/kp2d/scratch/69.ckpt",
                "conf_threshold": 0.0,
                "k_best": 1000
            }
            
        },
        
        "kp2d_2": {
            
            "params": {
                "model_path": "/mnt/g27prist/TCO/TCO-Studenten/ariel_research_project/pretrained_models/kp2d/scratch/69.ckpt",
                "conf_threshold": 0.5,
                "k_best": 1000
            }
            
        },
        
        "kp2d_3": {
            "params": {
                "model_path": "/mnt/g27prist/TCO/TCO-Studenten/ariel_research_project/pretrained_models/kp2d/scratch/69.ckpt",
                "conf_threshold": 0.7,
                "k_best": 1000
            }
        }
    }
    
    IMGS_PATH = "/mnt/g27prist/TCO/TCO-Studenten/ariel_research_project/raw_dataset_videos/testing/dataset/"
    OUTPUT_DIRECTORY = "/mnt/g27prist/TCO/TCO-Studenten/ariel_research_project/raw_dataset_videos/testing/results/scratch_trained_methods_outputs_2/"
    METHOD_NAME = "KP2D"
    IMGS_EXTENSION = "ppm"
    NUMBER_OF_IMAGES = 5  # Including the original
    N_JOBS = 1

    execute_grid(GRID, IMGS_PATH, OUTPUT_DIRECTORY, METHOD_NAME, IMGS_EXTENSION, NUMBER_OF_IMAGES, N_JOBS)
