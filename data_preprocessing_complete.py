# data_preprocessing_complete.py
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml
import json  # 添加这一行导入 json 模块


class DataPreprocessor:
    def __init__(self, dataset_path='coco_cup_dataset'):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, 'images')
        self.labels_dir = os.path.join(dataset_path, 'labels')

    def analyze_dataset(self):
        """分析数据集统计信息"""
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        label_files = [f for f in os.listdir(self.labels_dir) if f.endswith('.txt')]

        print(f"数据集分析结果:")
        print(f"图像数量: {len(image_files)}")
        print(f"标注数量: {len(label_files)}")

        # 分析标注统计
        bbox_sizes = []
        bbox_counts = []

        for label_file in label_files[:100]:  # 抽样分析
            label_path = os.path.join(self.labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                bbox_counts.append(len(lines))

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        bbox_sizes.append(width * height)

        if bbox_sizes:
            print(f"平均边界框数量: {np.mean(bbox_counts):.2f}")
            print(f"平均边界框大小: {np.mean(bbox_sizes):.4f}")
            print(f"边界框大小标准差: {np.std(bbox_sizes):.4f}")

        return len(image_files), len(label_files)

    def visualize_samples(self, num_samples=5):
        """可视化数据样本"""
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

        if not image_files:
            print("没有找到图像文件")
            return

        # 创建可视化目录
        viz_dir = os.path.join(self.dataset_path, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)

        for i in range(min(num_samples, len(image_files))):
            img_file = image_files[i]
            img_path = os.path.join(self.images_dir, img_file)
            label_path = os.path.join(self.labels_dir, img_file.replace('.jpg', '.txt'))

            # 读取图像
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w = img.shape[:2]

            # 读取标注
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # 绘制边界框
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)

                        # 转换为像素坐标
                        x1 = int((x_center - width / 2) * img_w)
                        y1 = int((y_center - height / 2) * img_h)
                        x2 = int((x_center + width / 2) * img_w)
                        y2 = int((y_center + height / 2) * img_h)

                        # 绘制矩形
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, 'Cup', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 保存可视化结果
            output_path = os.path.join(viz_dir, f'sample_{i + 1}.png')
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f'Sample {i + 1}: {img_file}')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()

        print(f"样本可视化完成，保存至: {viz_dir}")

    def create_augmentation_pipeline(self):
        """创建数据增强流水线"""
        # 基础增强
        basic_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # 高级增强
        advanced_transform = A.Compose([
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        return basic_transform, advanced_transform

    def apply_augmentation(self, image, bboxes, class_labels, transform):
        """应用数据增强"""
        if len(bboxes) == 0:
            return image, bboxes, class_labels

        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        return transformed['image'], transformed['bboxes'], transformed['class_labels']

    def split_dataset(self, test_size=0.2, val_size=0.1):
        """划分训练集、验证集和测试集"""
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

        # 第一次划分：分离测试集
        train_val_files, test_files = train_test_split(
            image_files, test_size=test_size, random_state=42
        )

        # 第二次划分：分离验证集
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size / (1 - test_size), random_state=42
        )

        # 创建划分目录
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, files in splits.items():
            split_dir = os.path.join(self.dataset_path, split_name)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

            print(f"{split_name}集: {len(files)} 张图像")

            # 创建文件列表
            with open(os.path.join(self.dataset_path, f'{split_name}.txt'), 'w') as f:
                for img_file in files:
                    f.write(f'./images/{img_file}\n')

        return splits

    def create_final_dataset_config(self, splits):
        """创建最终的数据集配置文件"""
        config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'train.txt',
            'val': 'val.txt',
            'test': 'test.txt',
            'nc': 1,
            'names': ['cup'],
            'download': None
        }

        with open(os.path.join(self.dataset_path, 'dataset.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print("最终数据集配置文件创建完成")

        # 保存数据集统计信息
        stats = {
            'total_images': sum(len(files) for files in splits.values()),
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
            'class_distribution': {'cup': 1},
            'creation_date': str(np.datetime64('now'))
        }

        with open(os.path.join(self.dataset_path, 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        print("数据集统计信息保存完成")

    def run_complete_preprocessing(self):
        """运行完整的数据预处理流程"""
        print("开始完整的数据预处理流程...")

        # 1. 分析数据集
        print("\n1. 分析数据集...")
        image_count, label_count = self.analyze_dataset()

        if image_count != label_count:
            print("警告: 图像和标注数量不匹配!")
            return False

        # 2. 可视化样本
        print("\n2. 可视化数据样本...")
        self.visualize_samples(num_samples=8)

        # 3. 划分数据集
        print("\n3. 划分数据集...")
        splits = self.split_dataset(test_size=0.2, val_size=0.1)

        # 4. 创建最终配置
        print("\n4. 创建数据集配置...")
        self.create_final_dataset_config(splits)

        print("\n数据预处理完成!")
        return True


def main():
    """主函数"""
    preprocessor = DataPreprocessor()
    success = preprocessor.run_complete_preprocessing()

    if success:
        print("\n✅ 数据预处理成功完成!")
    else:
        print("\n❌ 数据预处理失败!")


if __name__ == "__main__":
    main()