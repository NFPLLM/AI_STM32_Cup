# training_targets_complete.py
import numpy as np
import json
from datetime import datetime


class TrainingTargets:
    def __init__(self):
        self.targets = self.define_targets()
        self.results = {}

    def define_targets(self):
        """定义完整的训练目标体系"""
        return {
            'accuracy_targets': {
                'mAP50': {'min': 0.35, 'target': 0.40, 'max': 0.50},
                'precision': {'min': 0.50, 'target': 0.60, 'max': 0.70},
                'recall': {'min': 0.30, 'target': 0.35, 'max': 0.45},
                'training_loss': {'min': 1.0, 'target': 1.5, 'max': 2.0}
            },
            'model_size_targets': {
                'original_size_mb': 25.6,
                'max_size_mb': 15.0,
                'target_size_mb': 10.0,
                'optimized_size_mb': 8.0
            },
            'performance_targets': {
                'pc_inference_ms': {'min': 50, 'target': 100, 'max': 200},
                'stm32_inference_ms': {'min': 300, 'target': 500, 'max': 800},
                'min_fps': {'min': 1.0, 'target': 2.0, 'max': 3.0},
                'memory_usage_kb': {'min': 256, 'target': 512, 'max': 1024}
            },
            'resource_targets': {
                'ram_usage_kb': 512,
                'flash_usage_kb': 1024,
                'power_consumption_ma': 150,
                'temperature_c': 65
            }
        }

    def validate_targets(self, actual_results):
        """验证目标达成情况"""
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'detailed_results': {},
            'recommendations': []
        }

        total_score = 0
        max_score = 0

        # 验证精度目标
        accuracy_score = self.validate_accuracy(actual_results.get('accuracy', {}))
        validation_report['detailed_results']['accuracy'] = accuracy_score
        total_score += accuracy_score['score']
        max_score += accuracy_score['max_score']

        # 验证模型大小目标
        size_score = self.validate_model_size(actual_results.get('model_size', {}))
        validation_report['detailed_results']['model_size'] = size_score
        total_score += size_score['score']
        max_score += size_score['max_score']

        # 验证性能目标
        performance_score = self.validate_performance(actual_results.get('performance', {}))
        validation_report['detailed_results']['performance'] = performance_score
        total_score += performance_score['score']
        max_score += performance_score['max_score']

        # 计算总体得分
        if max_score > 0:
            validation_report['overall_score'] = (total_score / max_score) * 100

        # 生成改进建议
        validation_report['recommendations'] = self.generate_recommendations(validation_report)

        return validation_report

    def validate_accuracy(self, accuracy_results):
        """验证精度目标"""
        score = 0
        max_score = 3  # 三个主要精度指标

        targets = self.targets['accuracy_targets']

        if 'mAP50' in accuracy_results:
            map50 = accuracy_results['mAP50']
            target = targets['mAP50']
            if map50 >= target['target']:
                score += 1
                print(f"✅ mAP50达标: {map50:.4f} >= {target['target']}")
            elif map50 >= target['min']:
                score += 0.5
                print(f"⚠️ mAP50基本达标: {map50:.4f}")
            else:
                print(f"❌ mAP50未达标: {map50:.4f} < {target['min']}")

        if 'precision' in accuracy_results:
            precision = accuracy_results['precision']
            target = targets['precision']
            if precision >= target['target']:
                score += 1
                print(f"✅ 精确率达标: {precision:.4f} >= {target['target']}")
            elif precision >= target['min']:
                score += 0.5
                print(f"⚠️ 精确率基本达标: {precision:.4f}")
            else:
                print(f"❌ 精确率未达标: {precision:.4f} < {target['min']}")

        if 'recall' in accuracy_results:
            recall = accuracy_results['recall']
            target = targets['recall']
            if recall >= target['target']:
                score += 1
                print(f"✅ 召回率达标: {recall:.4f} >= {target['target']}")
            elif recall >= target['min']:
                score += 0.5
                print(f"⚠️ 召回率基本达标: {recall:.4f}")
            else:
                print(f"❌ 召回率未达标: {recall:.4f} < {target['min']}")

        return {'score': score, 'max_score': max_score}

    def validate_model_size(self, size_results):
        """验证模型大小目标"""
        score = 0
        max_score = 1

        if 'model_size_mb' in size_results:
            size = size_results['model_size_mb']
            target = self.targets['model_size_targets']

            if size <= target['optimized_size_mb']:
                score = 1
                print(f"✅ 模型大小优秀: {size:.1f}MB <= {target['optimized_size_mb']}MB")
            elif size <= target['target_size_mb']:
                score = 0.8
                print(f"✅ 模型大小达标: {size:.1f}MB <= {target['target_size_mb']}MB")
            elif size <= target['max_size_mb']:
                score = 0.5
                print(f"⚠️ 模型大小基本达标: {size:.1f}MB <= {target['max_size_mb']}MB")
            else:
                print(f"❌ 模型大小超标: {size:.1f}MB > {target['max_size_mb']}MB")

        return {'score': score, 'max_score': max_score}

    def validate_performance(self, performance_results):
        """验证性能目标"""
        score = 0
        max_score = 2

        targets = self.targets['performance_targets']

        if 'inference_time_ms' in performance_results:
            inference_time = performance_results['inference_time_ms']
            target = targets['stm32_inference_ms']

            if inference_time <= target['target']:
                score += 1
                print(f"✅ 推理时间达标: {inference_time}ms <= {target['target']}ms")
            elif inference_time <= target['max']:
                score += 0.5
                print(f"⚠️ 推理时间基本达标: {inference_time}ms <= {target['max']}ms")
            else:
                print(f"❌ 推理时间超标: {inference_time}ms > {target['max']}ms")

        if 'fps' in performance_results:
            fps = performance_results['fps']
            target = targets['min_fps']

            if fps >= target['target']:
                score += 1
                print(f"✅ 帧率达标: {fps}FPS >= {target['target']}FPS")
            elif fps >= target['min']:
                score += 0.5
                print(f"⚠️ 帧率基本达标: {fps}FPS >= {target['min']}FPS")
            else:
                print(f"❌ 帧率未达标: {fps}FPS < {target['min']}FPS")

        return {'score': score, 'max_score': max_score}

    def generate_recommendations(self, validation_report):
        """生成改进建议"""
        recommendations = []
        results = validation_report['detailed_results']

        # 精度改进建议
        if 'accuracy' in results and results['accuracy']['score'] < 2:
            recommendations.extend([
                "增加数据增强强度",
                "调整学习率策略",
                "尝试不同的优化器",
                "增加训练轮数"
            ])

        # 模型大小改进建议
        if 'model_size' in results and results['model_size']['score'] < 0.8:
            recommendations.extend([
                "使用模型剪枝技术",
                "尝试更小的输入尺寸",
                "减少网络层数",
                "使用通道剪枝"
            ])

        # 性能改进建议
        if 'performance' in results and results['performance']['score'] < 1.5:
            recommendations.extend([
                "优化图像预处理流程",
                "使用INT8量化",
                "优化内存访问模式",
                "减少中间缓冲区"
            ])

        return recommendations

    def save_report(self, report, filename="training_targets_report.json"):
        """保存验证报告"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"验证报告已保存: {filename}")


# 使用示例
def example_usage():
    """目标验证使用示例"""
    targets = TrainingTargets()

    # 模拟实际结果
    actual_results = {
        'accuracy': {
            'mAP50': 0.4256,
            'precision': 0.6371,
            'recall': 0.3816
        },
        'model_size': {
            'model_size_mb': 11.5
        },
        'performance': {
            'inference_time_ms': 280,
            'fps': 3.5
        }
    }

    # 验证目标达成情况
    report = targets.validate_targets(actual_results)

    print(f"\n总体得分: {report['overall_score']:.1f}%")
    print("\n改进建议:")
    for i, recommendation in enumerate(report['recommendations'], 1):
        print(f"{i}. {recommendation}")

    # 保存报告
    targets.save_report(report)


if __name__ == "__main__":
    example_usage()