# model_validation_complete.py
import os
import cv2
import re  # 新增：用于解析模型信息字符串
import json  # 新增：用于生成JSON报告
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import onnxruntime as ort
from sklearn.metrics import precision_recall_curve, average_precision_score


class ModelValidator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.ort_session = None

    def load_model(self):
        """加载模型"""
        if self.model_path.endswith('.pt'):
            self.model = YOLO(self.model_path)
            print(f"PyTorch模型加载成功: {self.model_path}")
            return True
        elif self.model_path.endswith('.onnx'):
            self.ort_session = ort.InferenceSession(self.model_path)
            print(f"ONNX模型加载成功: {self.model_path}")
            return True
        else:
            print(f"不支持的模型格式: {self.model_path}")
            return False

    def validate_accuracy(self, data_config):
        """验证模型精度（修复：处理召回率数组问题）"""
        if self.model is None:
            print("PyTorch模型未加载")
            return None

        print("开始精度验证...")
        metrics = self.model.val(data=data_config, split='val')

        # 处理精确率：若为数组取第一个元素（所有类平均）
        precision = metrics.box.p[0] if isinstance(metrics.box.p, np.ndarray) else metrics.box.p
        # 处理召回率：同上（修复核心点）
        recall = metrics.box.r[0] if isinstance(metrics.box.r, np.ndarray) else metrics.box.r

        print("\n=== 精度验证结果 ===")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")

        return {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': precision,
            'recall': recall
        }

    def test_inference_speed(self, num_iterations=50):
        """测试推理速度（优化：处理动态输入形状）"""
        if self.ort_session is None:
            print("ONNX模型未加载")
            return None

        input_name = self.ort_session.get_inputs()[0].name
        input_shape = self.ort_session.get_inputs()[0].shape

        # 优化：处理动态维度（如[-1,3,-1,-1]），替换为固定尺寸(1,3,640,640)
        fixed_input_shape = []
        for dim in input_shape:
            if dim == '?' or dim < 1:  # ONNX动态维度可能用?或负数表示
                fixed_input_shape.append(640 if len(fixed_input_shape) in [2, 3] else 1)
            else:
                fixed_input_shape.append(dim)
        fixed_input_shape = tuple(fixed_input_shape)

        # 创建测试输入（使用固定形状）
        test_input = np.random.randn(*fixed_input_shape).astype(np.float32)

        # 预热
        for _ in range(10):
            self.ort_session.run(None, {input_name: test_input})

        # 正式测试
        import time
        times = []
        for i in range(num_iterations):
            start_time = time.perf_counter()
            self.ort_session.run(None, {input_name: test_input})
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

        # 分析结果
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print("\n=== 推理速度测试 ===")
        print(f"输入形状: {fixed_input_shape}")
        print(f"平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"最快推理时间: {min_time:.2f} ms")
        print(f"最慢推理时间: {max_time:.2f} ms")
        print(f"理论FPS: {1000 / avg_time:.2f}")

        # STM32性能预估
        stm32_factor = 15  # STM32比PC慢的倍数（可根据实际硬件调整）
        stm32_avg_time = avg_time * stm32_factor
        stm32_fps = 1000 / stm32_avg_time

        print(f"\nSTM32性能预估:")
        print(f"预估推理时间: {stm32_avg_time:.1f} ms")
        print(f"预估FPS: {stm32_fps:.1f}")

        return {
            'pc_avg_time_ms': avg_time,
            'pc_fps': 1000 / avg_time,
            'stm32_avg_time_ms': stm32_avg_time,
            'stm32_fps': stm32_fps
        }

    def visualize_detections(self, image_paths, output_dir='validation_results'):
        """可视化检测结果"""
        if self.model is None:
            print("PyTorch模型未加载")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"可视化检测结果，保存至: {output_dir}")

        for i, image_path in enumerate(image_paths):
            if not os.path.exists(image_path):
                print(f"图像不存在: {image_path}")
                continue

            # 进行推理（设置conf=0.3提高检测召回率）
            results = self.model(image_path, conf=0.3)

            # 绘制并保存结果
            for r in results:
                im_array = r.plot()  # YOLO内置可视化
                output_path = os.path.join(output_dir, f'detection_{i + 1}.png')
                cv2.imwrite(output_path, im_array)
                print(f"保存检测结果: {output_path}")

    def analyze_model_complexity(self):
        """分析模型复杂度（修复：处理元组格式的model.info()返回值）"""
        if self.model is None:
            print("PyTorch模型未加载")
            return None

        # 获取模型信息（元组格式：(层数, 参数量, 梯度数, GFLOPs)）
        model_info_tuple = self.model.info()
        print("\n模型原始信息（元组）:")
        print(model_info_tuple)

        # 从元组中提取信息（按固定位置解析）
        layers = model_info_tuple[0] if len(model_info_tuple) > 0 else None
        params = model_info_tuple[1] if len(model_info_tuple) > 1 else None
        gradients = model_info_tuple[2] if len(model_info_tuple) > 2 else None
        gflops = model_info_tuple[3] if len(model_info_tuple) > 3 else None

        # 计算模型文件大小
        model_size_mb = os.path.getsize(self.model_path) / 1024 / 1024 if os.path.exists(self.model_path) else None

        # 打印分析结果
        print("\n=== 模型复杂度分析 ===")
        print(f"层数: {layers}" if layers is not None else "层数: 无法获取")
        print(f"参数量: {params:,}" if params is not None else "参数量: 无法获取")
        print(f"梯度数量: {gradients:,}" if gradients is not None else "梯度数量: 无法获取")
        print(f"计算量: {gflops:.2f} GFLOPs" if gflops is not None else "计算量: 无法获取")
        print(f"模型大小: {model_size_mb:.2f} MB" if model_size_mb else "模型大小: 无法获取")

        # 返回结构化信息
        return {
            'layers': layers,
            'parameters': params,
            'gradients': gradients,
            'gflops': gflops,
            'model_size_mb': model_size_mb
        }

    def generate_validation_report(self, accuracy_results, speed_results, output_file='validation_report.json'):
        """生成验证报告（修复：处理speed_results为None的情况）"""
        # 初始化报告基础信息
        report = {
            'validation_date': str(np.datetime64('now')),
            'model_path': self.model_path,
            'accuracy_metrics': accuracy_results or {},  # 若为None则设为空字典
            'performance_metrics': speed_results or {},  # 若为None则设为空字典
            'hardware_compatibility': {
                'recommended_ram': '512KB',
                'recommended_flash': '8MB',
                'suitable_for_stm32': False,
                'performance_level': 'Unknown'
            },
            'recommendations': []
        }

        # 修复：仅当speed_results存在时，判断STM32兼容性
        if speed_results and 'stm32_fps' in speed_results:
            stm32_fps = speed_results['stm32_fps']
            report['hardware_compatibility']['suitable_for_stm32'] = stm32_fps >= 2.0
            report['hardware_compatibility']['performance_level'] = 'Good' if stm32_fps >= 2.0 else 'Limited'

        # 生成优化建议
        if accuracy_results and accuracy_results.get('mAP50', 0) < 0.4:
            report['recommendations'].append("考虑增加训练数据或调整训练参数（如增大epoch、调整学习率）")
        if speed_results and speed_results.get('stm32_fps', 0) < 2.0:
            report['recommendations'].append("建议使用更小的输入尺寸（如416x416）或模型量化（如INT8）")
        if not report['recommendations']:
            report['recommendations'].append("模型性能满足基础需求，可根据实际场景进一步优化")

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n验证报告已保存: {output_file}")
        return report

    def run_complete_validation(self, data_config, test_images=None):
        """运行完整的验证流程"""
        print("=" * 50)
        print("开始完整的模型验证流程...")
        print("=" * 50)

        # 1. 加载模型
        if not self.load_model():
            print("模型加载失败，终止验证流程")
            return False

        # 2. 验证精度（仅PyTorch模型支持）
        print("\n1. 验证模型精度...")
        accuracy_results = self.validate_accuracy(data_config)

        # 3. 分析模型复杂度
        print("\n2. 分析模型复杂度...")
        complexity_results = self.analyze_model_complexity()

        # 4. 测试推理速度（仅ONNX模型支持）
        print("\n3. 测试推理速度...")
        speed_results = self.test_inference_speed()

        # 5. 可视化检测结果（可选）
        if test_images:
            print("\n4. 可视化检测结果...")
            self.visualize_detections(test_images)
        else:
            print("\n4. 可视化检测结果... 未提供测试图片路径，跳过此步骤")

        # 6. 生成验证报告
        print("\n5. 生成验证报告...")
        report = self.generate_validation_report(accuracy_results, speed_results)

        print("\n" + "=" * 50)
        print("✅ 完整验证流程完成!")
        print("=" * 50)
        return report


def main():
    """主函数（可根据实际路径调整）"""
    # 1. 配置路径（请确保这两个路径正确）
    model_path = 'cup_detection/yolov8n_cup_improved/weights/best.pt'  # .pt或.onnx模型路径
    data_config = 'coco_cup_dataset/dataset.yaml'  # 数据集配置文件路径
    # 可选：测试图片路径（用于可视化），若不需要可设为None
    test_images = [
        # "test_images/cup1.jpg",
        # "test_images/cup2.jpg"
    ]

    # 2. 初始化验证器并运行
    validator = ModelValidator(model_path)
    report = validator.run_complete_validation(data_config, test_images)

    # 3. 打印报告摘要
    if report and report.get('accuracy_metrics'):
        print("\n📊 验证报告摘要:")
        acc = report['accuracy_metrics']
        print(f"精度 - mAP50: {acc.get('mAP50', 0):.4f}, 精确率: {acc.get('precision', 0):.4f}, 召回率: {acc.get('recall', 0):.4f}")
        if report.get('performance_metrics'):
            perf = report['performance_metrics']
            print(f"性能 - PC FPS: {perf.get('pc_fps', 0):.1f}, STM32预估FPS: {perf.get('stm32_fps', 0):.1f}")
        print(f"硬件适配 - 适合STM32: {report['hardware_compatibility']['suitable_for_stm32']}")


if __name__ == "__main__":
    main()