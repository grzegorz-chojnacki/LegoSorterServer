import argparse
import time
import logging

from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

from lego_sorter_server.analysis.AnalysisService import AnalysisService
from lego_sorter_server.analysis.detection.LegoLabeler import LegoLabeler


def is_image(file_name):
    extension = file_name.split('.')[-1]
    return extension == 'jpeg' or extension == 'jpg' or extension == 'png'


def process_images_in_path(input_path: Path, output_path: Path, analysis_service: AnalysisService, skip_images: bool):
    start_time = time.time()
    counter = 0

    labeler = LegoLabeler()

    for file in input_path.iterdir():
        if file.is_file() and is_image(file.name):
            xml_name = file.name.split(".")[0] + ".xml"
            dest_path_img = output_path / file.name
            dest_path_xml = output_path / xml_name
            image = Image.open(file)
            detection_results = analysis_service.detect(image, threshold=0.8, discard_border_results=False)
            width, height = image.size
            label_file = labeler.to_label_file(file.name, dest_path_xml, width, height, detection_results.detection_boxes)
            with open(dest_path_xml, "w") as label_xml:
                label_xml.write(label_file)
            if not skip_images:
                image.save(dest_path_img)
            counter += 1

    seconds_elapsed = time.time() - start_time
    print(
        f"Processing path {input_path} took {seconds_elapsed} seconds, "
        f"{1000 * (seconds_elapsed / counter) if counter != 0 else 0} ms per image."
    )


def process_recursive(input_path: Path,
                      output_path: Path,
                      executor: ThreadPoolExecutor,
                      analysis_service: AnalysisService,
                      skip_images: bool):
    output_path.mkdir(exist_ok=True)
    dirs_to_process = []

    for file in input_path.iterdir():
        if file.is_dir():
            dirs_to_process.append(file)

    futures = []
    for directory in dirs_to_process:
        sub_out_path = (output_path / directory.name)
        futures += process_recursive(directory, sub_out_path, executor, analysis_service, skip_images)

    futures.append(executor.submit(process_images_in_path, input_path, output_path, analysis_service, skip_images))
    return futures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detects lego bricks on images. This script copies the input directory structure.')
    parser.add_argument('-i' '--input_dir', required=True, help='A path to a directory containing images to process.',
                        type=str, dest='input')
    parser.add_argument('-o', '--output_path', required=True, help='An output path.', type=str, dest='output')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Process images in the input_path and its subdirectories.')
    parser.add_argument('-si', '--skip_images', action='store_true', dest='skip_images',
                        help='Whether to skip copying images from the input directory to the output directory.')
    args = parser.parse_args()

    logging.getLogger().disabled = True
    analysis_service = AnalysisService()

    if args.recursive:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = process_recursive(Path(args.input), Path(args.output), executor, analysis_service,
                                        args.skip_images)
            for future in futures:
                future.result()
    else:
        process_images_in_path(Path(args.input), Path(args.output), analysis_service, args.skip_images)
