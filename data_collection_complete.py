# data_collection_complete.py
import os
import requests
import zipfile
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2
import numpy as np


class CocoDataCollector:
    def __init__(self):
        self.coco = None
        self.dataset_path = 'coco_cup_dataset'

    def setup_directories(self):
        """创建数据目录结构"""
        directories = [
            f'{self.dataset_path}/images',
            f'{self.dataset_path}/labels',
            f'{self.dataset_path}/backups'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("目录结构创建完成")

    def download_annotations(self):
        """下载COCO标注文件"""
        annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        local_zip_path = "annotations_trainval2017.zip"

        if os.path.exists(local_zip_path):
            print("标注文件已存在，跳过下载")
            return True

        print("开始下载COCO标注文件...")
        try:
            response = requests.get(annotation_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(local_zip_path, 'wb') as f, tqdm(
                    desc="下载进度",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)

            # 解压文件
            print("解压标注文件...")
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")

            return True

        except Exception as e:
            print(f"下载失败: {e}")
            return False

    def initialize_coco_api(self):
        """初始化COCO API"""
        annotation_file = 'annotations/instances_train2017.json'
        if not os.path.exists(annotation_file):
            print("标注文件不存在")
            return False

        self.coco = COCO(annotation_file)
        print("COCO API初始化成功")
        return True

    def get_cup_images(self, max_images=500):
        """获取杯子图像ID"""
        cat_ids = self.coco.getCatIds(catNms=['cup'])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        print(f"找到 {len(img_ids)} 张包含杯子的图像")

        # 筛选高质量图像
        filtered_ids = []
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
            anns = self.coco.loadAnns(ann_ids)

            # 只保留有清晰标注的图像
            valid_annotations = 0
            for ann in anns:
                if ann['area'] > 1000 and not ann['iscrowd']:
                    valid_annotations += 1

            if valid_annotations >= 1:
                filtered_ids.append(img_id)

            if len(filtered_ids) >= max_images:
                break

        print(f"筛选后保留 {len(filtered_ids)} 张图像")
        return filtered_ids

    def download_images(self, img_ids):
        """下载图像文件"""
        downloaded_count = 0
        failed_count = 0

        print("开始下载图像...")
        for img_id in tqdm(img_ids, desc="下载图像"):
            try:
                img_info = self.coco.loadImgs(img_id)[0]
                img_url = img_info['coco_url']

                # 下载图像
                response = requests.get(img_url, timeout=30)
                if response.status_code == 200:
                    img_path = f'{self.dataset_path}/images/{img_info["file_name"]}'

                    with open(img_path, 'wb') as f:
                        f.write(response.content)

                    # 创建标注文件
                    self.create_annotation_file(img_info)
                    downloaded_count += 1

                else:
                    failed_count += 1

            except Exception as e:
                print(f"下载图像 {img_id} 失败: {e}")
                failed_count += 1
                continue

        print(f"下载完成! 成功: {downloaded_count}, 失败: {failed_count}")
        return downloaded_count

    def create_annotation_file(self, img_info):
        """创建YOLO格式标注文件"""
        cat_ids = self.coco.getCatIds(catNms=['cup'])
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        txt_path = f'{self.dataset_path}/labels/{img_info["file_name"].replace(".jpg", ".txt")}'

        with open(txt_path, 'w') as f:
            for ann in anns:
                if ann['category_id'] in cat_ids and not ann['iscrowd']:
                    # COCO格式转YOLO格式
                    bbox = ann['bbox']  # [x, y, width, height]
                    image_width = img_info['width']
                    image_height = img_info['height']

                    # 转换为YOLO格式: [x_center, y_center, width, height] 相对坐标
                    x_center = (bbox[0] + bbox[2] / 2) / image_width
                    y_center = (bbox[1] + bbox[3] / 2) / image_height
                    width = bbox[2] / image_width
                    height = bbox[3] / image_height

                    # 写入标注文件
                    f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

    def validate_dataset(self):
        """验证数据集完整性"""
        images_dir = f'{self.dataset_path}/images'
        labels_dir = f'{self.dataset_path}/labels'

        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        print(f"\n数据集验证结果:")
        print(f"图像数量: {len(images)}")
        print(f"标注数量: {len(labels)}")

        # 检查标注文件内容
        if labels:
            sample_label = os.path.join(labels_dir, labels[0])
            with open(sample_label, 'r') as f:
                content = f.read().strip()
            print(f"示例标注: {content}")

        return len(images) == len(labels)

    def create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'images',
            'val': 'images',
            'nc': 1,
            'names': ['cup'],
            'download': None
        }

        with open(f'{self.dataset_path}/data.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print("数据集配置文件创建完成")


def main():
    """主函数"""
    collector = CocoDataCollector()

    # 设置目录
    collector.setup_directories()

    # 下载标注文件
    if not collector.download_annotations():
        return

    # 初始化COCO API
    if not collector.initialize_coco_api():
        return

    # 获取图像ID
    img_ids = collector.get_cup_images(max_images=500)

    # 下载图像
    downloaded_count = collector.download_images(img_ids)

    # 验证数据集
    collector.validate_dataset()

    # 创建配置文件
    collector.create_dataset_config()

    print(f"\n数据收集完成! 共处理 {downloaded_count} 张图像")


if __name__ == "__main__":
    main()