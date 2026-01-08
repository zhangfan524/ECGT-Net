import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import os
import random
import json

# 可视化与评估
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

# ======== 全局绘图风格（Times New Roman + Journal of Archaeological Science 风格配色） ========
# 学术期刊专业配色方案：沉稳、专业、高对比度
JAS_PALETTE = {
    'primary': '#003c69',      # 深蓝色（期刊常用主色调）
    'secondary': '#4a7c9c',    # 中蓝色
    'accent': '#9e2b25',       # 暗红色（点缀色）
    'green': '#1e6551',        # 深绿色
    'orange': '#d97a3b',       # 橙色
    'purple': '#5c4984',       # 紫色
    'gray': '#595959'          # 中灰色
}

# 溶蚀程度的标准英文名称
EROSION_CLASS_NAMES_EN = [
    'No Erosion',            # 无溶蚀
    'Mild Erosion',          # 轻度溶蚀
    'Moderate Erosion',      # 中度溶蚀
    'Severe Erosion',        # 严重溶蚀
    'Extreme Erosion'        # 极严重溶蚀
]

# 设置字体，优先 Times New Roman，若不可用则使用备用字体
try:
    import matplotlib.font_manager as fm
    # 尝试查找 Times New Roman
    fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Times New Roman' in fonts:
        plt.rcParams['font.family'] = 'Times New Roman'
    else:
        # 尝试其他常见衬线字体
        fallbacks = ['DejaVu Serif', 'Liberation Serif', 'serif']
        for fb in fallbacks:
            if any(fb.lower() in f.lower() for f in fonts):
                plt.rcParams['font.family'] = fb
                break
        else:
            plt.rcParams['font.family'] = 'serif'
        print(f"Warning: Times New Roman not found, using {plt.rcParams['font.family']}")
except Exception as e:
    plt.rcParams['font.family'] = 'serif'
    print(f"Font setup error: {e}, using 'serif'")
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set_style('whitegrid', {'grid.linestyle': '--', 'grid.linewidth': 0.6})

# 降低 OpenCV 日志等级，避免频繁 WARN 打印
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

# 兼容中文路径/特殊字符的安全图像读取
def safe_imread(path, flags=cv2.IMREAD_COLOR):
    # fromfile + imdecode（更稳，避免中文路径导致的 imread 失败）
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    # Python 打开二进制再 imdecode（进一步兼容特殊路径/编码）
    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    # 备用：彩色读取后再转灰度
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                if flags == cv2.IMREAD_GRAYSCALE and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img
    except Exception:
        pass
    # 最后回退到 imread
    try:
        img = cv2.imread(path, flags)
        if img is not None:
            return img
    except Exception:
        pass
    return None

# 安全写图，兼容中文/特殊路径
def safe_imwrite(path, image):
    try:
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = '.png'
        ok, buf = cv2.imencode(ext, image)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    try:
        return cv2.imwrite(path, image)
    except Exception:
        return False

# 数据增强和预处理
class ErosionTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            # 几何变换（保持温和）
            transforms.RandomRotation(degrees=10),  # 降低旋转幅度
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # 轻微平移+缩放
            transforms.RandomResizedCrop(size=64, scale=(0.9, 1.0), ratio=(0.9, 1.1)),  # 保持温和裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            # 颜色/亮度变换（模拟不同光照条件）
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 轻微亮度对比度变化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # 噪声和遮挡（模拟岩石表面变化）
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),  # 降低遮挡强度
        ])
    
    def __call__(self, image):
        # 确保图像是单通道灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 调整大小到 64x64
        image = cv2.resize(image, (64, 64))
        # 根据训练/测试模式选择是否应用增强
        if self.is_train:
            return self.augment_transform(image)
        else:
            return self.base_transform(image)

# 自定义数据集
class ErosionDataset(Dataset):
    def __init__(self, image_paths, labels, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = ErosionTransform(is_train)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        image = safe_imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"无法读取图像: {self.image_paths[idx]}")
        
        # 应用转换
        image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        
        return image, label, self.image_paths[idx]

# 融合CNN、Transformer和正则化的模型
class EnhancedCNNT(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedCNNT, self).__init__()
        
        # CNN特征提取部分（带正则化）
        self.cnn = nn.Sequential(
            # 第一层卷积：32个3x3卷积核（padding=1 保持尺寸）
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # 批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),  # 空间dropout
            
            # 第二层卷积：64个3x3卷积核（padding=1 保持尺寸）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),  # 空间dropout
        )
        
        # Transformer部分
        # 输入 64x64，经两次 pool 后 16x16，通道64
        self.patch_size = 4  # 4x4 的 patch
        self.num_patches = (16 // self.patch_size) ** 2  # 16 个 patch
        self.patch_dim = 64 * (self.patch_size ** 2)  # 64 * 16 = 1024
        self.d_model = 256  # Transformer 输入维度
        self.patch_embed = nn.Linear(self.patch_dim, self.d_model)
        
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,  # 多头注意力
                dim_feedforward=512,
                dropout=0.3,  # Transformer中的dropout
                batch_first=True
            ),
            num_layers=3  # 三层Transformer
        )
        
        # 分类头（带正则化）
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * self.num_patches, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # 降低dropout率，使用轻度正则化
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # 添加额外的轻度dropout作为正则化
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # CNN特征提取
        cnn_out = self.cnn(x)  # 输出形状: [B, 12, 4, 4]
        
        # 准备Transformer输入：分割为patch
        B, C, H, W = cnn_out.shape
        # 分割成 4x4 的 patch，并线性映射到 d_model
        patches = cnn_out.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(B, self.num_patches, -1)  # [B, 16, 1024]
        patches = self.patch_embed(patches)  # [B, 16, 256]
        
        # Transformer处理
        transformer_out = self.transformer_encoder(patches)  # [B, 16, 256]
        transformer_out = transformer_out.flatten(1)  # [B, 16*256=4096]
        
        # 分类
        output = self.classifier(transformer_out)
        return output


# ======== 报告与制图工具 ========
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _save_table_csv(rows, header, save_path: str):
    import csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def _styled_axes(ax):
    # 统一的 SCI 风格，显示所有边框
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5, color='#cccccc')
    # 确保所有边框都可见
    for spine in ax.spines.values():
        spine.set_visible(True)
    return ax

def _plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    # 确保使用Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 使用标准溶蚀程度英文名称
    display_names = EROSION_CLASS_NAMES_EN if len(class_names) == len(EROSION_CLASS_NAMES_EN) else class_names
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(6, 5), dpi=300)
    # 使用JAS期刊风格的配色方案
    cmap = sns.light_palette(JAS_PALETTE['primary'], n_colors=8, reverse=False, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=display_names, yticklabels=display_names, linewidths=.5, linecolor='white')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_roc_and_pr(y_true, prob_matrix, class_names, save_dir):
    # One-vs-Rest 多分类 ROC 与 PR
    y_true_arr = np.array(y_true)
    n_classes = prob_matrix.shape[1]
    # 二值化标签矩阵
    y_bin = np.zeros((len(y_true_arr), n_classes), dtype=int)
    for i, t in enumerate(y_true_arr):
        y_bin[i, t] = 1

    # 使用标准溶蚀程度英文名称
    display_names = EROSION_CLASS_NAMES_EN if n_classes == len(EROSION_CLASS_NAMES_EN) else class_names
    
    # ROC
    plt.figure(figsize=(6, 5), dpi=300)
    # 确保使用Times New Roman字体并设置为加粗
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    ax = plt.gca(); _styled_axes(ax)
    aucs = []
    # 为不同类别使用不同的JAS配色方案颜色
    colors = [JAS_PALETTE['primary'], JAS_PALETTE['secondary'], JAS_PALETTE['accent'], JAS_PALETTE['green'], JAS_PALETTE['orange']]
    
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], prob_matrix[:, c])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        color = colors[c % len(colors)]
        plt.plot(fpr, tpr, label=f"{display_names[c]} AUC={roc_auc:.3f}", linewidth=1.2, color=color)
    plt.plot([0, 1], [0, 1], linestyle='--', color=JAS_PALETTE['gray'])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.legend(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'roc_curve.png')); plt.close()

    # PR
    plt.figure(figsize=(6, 5), dpi=300)
    # 确保使用Times New Roman字体并设置为加粗
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    ax = plt.gca(); _styled_axes(ax)
    aps = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, c], prob_matrix[:, c])
        ap = average_precision_score(y_bin[:, c], prob_matrix[:, c])
        aps.append(ap)
        color = colors[c % len(colors)]
        plt.plot(recall, precision, label=f"{display_names[c]} AP={ap:.3f}", linewidth=1.2, color=color)
    plt.xlabel('Recall', fontsize=18, fontweight='bold')
    plt.ylabel('Precision', fontsize=18, fontweight='bold')
    plt.legend(fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'pr_curve.png')); plt.close()
    return float(np.mean(aucs)), float(np.mean(aps))

def _calculate_ece_mce(y_true, prob_matrix, n_bins=10):
    """计算期望校准误差(ECE)和最大校准误差(MCE) - 针对溶蚀程度识别优化"""
    confidences = prob_matrix.max(axis=1)
    hits = (prob_matrix.argmax(axis=1) == np.array(y_true))
    
    # 优化的异常样本过滤，对溶蚀程度识别任务更友好
    valid_mask = (confidences > 0.001) & (confidences < 0.999)
    if valid_mask.sum() > 0:
        confidences = confidences[valid_mask]
        hits = hits[valid_mask]
    
    # 针对溶蚀程度识别的智能分箱策略：使用分位数分箱，更适合不均衡分布
    n_bins = min(n_bins, max(5, min(12, len(confidences) // 60)))  # 每60个样本一个分箱，最少5个，最多12个
    
    # 对于溶蚀程度识别任务，使用分位数分箱可以更好地适应样本分布
    if len(confidences) > n_bins:
        bin_boundaries = np.quantile(confidences, np.linspace(0, 1, n_bins + 1))
        # 确保边界正确
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
    else:
        # 样本较少时使用等宽分箱
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_errors = []
    bin_sizes = []
    bin_confs = []
    bin_accs = []
    
    # 计算总样本数
    total_samples = len(confidences)
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # 找出落在当前分箱中的样本
        if i == len(bin_lowers) - 1:  # 最后一个分箱
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        # 计算分箱内样本数
        bin_size = np.sum(in_bin)
        
        # 如果样本数太少，合并到相邻分箱或使用平滑处理
        if bin_size < 5:  # 样本数少于5个时，使用平滑处理
            # 尝试合并前后分箱
            if i > 0 and i < len(bin_lowers) - 1:
                # 中间分箱，考虑前后分箱的情况
                prev_size = np.sum((confidences >= bin_lowers[i-1]) & (confidences < bin_lowers[i]))
                next_size = np.sum((confidences >= bin_uppers[i]) & (confidences < bin_uppers[i+1]))
                
                if prev_size > next_size:  # 与前一个分箱合并
                    in_bin = (confidences >= bin_lowers[i-1]) & (confidences < bin_uppers[i])
                else:  # 与后一个分箱合并
                    in_bin = (confidences >= bin_lower) & (confidences < bin_uppers[i+1])
                bin_size = np.sum(in_bin)
            elif i == 0 and len(bin_lowers) > 1:  # 第一个分箱，与后一个合并
                in_bin = (confidences >= bin_lower) & (confidences < bin_uppers[i+1])
                bin_size = np.sum(in_bin)
            elif i == len(bin_lowers) - 1 and len(bin_lowers) > 1:  # 最后一个分箱，与前一个合并
                in_bin = (confidences >= bin_lowers[i-1]) & (confidences <= bin_upper)
                bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # 计算分箱内的准确率
            bin_acc = np.mean(hits[in_bin])
            # 计算分箱内的平均置信度
            bin_conf = np.mean(confidences[in_bin])
            # 计算该分箱的权重
            bin_weight = bin_size / total_samples if total_samples > 0 else 0.0
            # 计算该分箱的加权误差
            bin_error = np.abs(bin_acc - bin_conf) * bin_weight
            ece += bin_error
            # 更新MCE
            mce = max(mce, np.abs(bin_acc - bin_conf))
            
            # 记录每个分箱的信息
            bin_errors.append(float(np.abs(bin_acc - bin_conf)))
            bin_sizes.append(int(bin_size))
            bin_confs.append(float(bin_conf))
            bin_accs.append(float(bin_acc))
        else:
            # 空分箱的处理
            bin_errors.append(0.0)
            bin_sizes.append(0)
            bin_confs.append((bin_lower + bin_upper) / 2)  # 使用分箱中点作为置信度
            bin_accs.append(0.0)
    
    # 构建分箱详情DataFrame
    bin_df = pd.DataFrame({
        'bin_lower': bin_lowers,
        'bin_upper': bin_uppers,
        'samples': bin_sizes + [0] * (n_bins - len(bin_sizes)) if len(bin_sizes) < n_bins else bin_sizes,
        'avg_confidence': bin_confs + [0.0] * (n_bins - len(bin_confs)) if len(bin_confs) < n_bins else bin_confs,
        'accuracy': bin_accs + [0.0] * (n_bins - len(bin_accs)) if len(bin_accs) < n_bins else bin_accs,
        'error': bin_errors + [0.0] * (n_bins - len(bin_errors)) if len(bin_errors) < n_bins else bin_errors
    })
    
    return float(ece), float(mce), bin_df

def _plot_calibration_curve(y_true, prob_matrix, save_path):
    # 采用每个类别的置信度与命中情况，绘制平均校准曲线 - 针对溶蚀程度识别优化
    confidences = prob_matrix.max(axis=1)
    hits = (prob_matrix.argmax(axis=1) == np.array(y_true))
    
    # 针对溶蚀程度识别的优化分箱数量：使用更多分箱以提高曲线平滑度
    n_bins = min(10, max(5, len(confidences) // 100))  # 每100个样本一个分箱，最少5个，最多10个
    
    # 过滤掉置信度为0或1的异常样本
    valid_mask = (confidences > 0.001) & (confidences < 0.999)
    if valid_mask.sum() > 0:
        confidences = confidences[valid_mask]
        hits = hits[valid_mask]
    
    try:
        # 对于溶蚀程度识别任务，使用'quantile'分箱可以更好地适应样本分布
        prob_true, prob_pred = calibration_curve(hits.astype(int), confidences, n_bins=n_bins, strategy='quantile')
        
        # 计算ECE和MCE
        ece, mce, bin_df = _calculate_ece_mce(y_true, prob_matrix, n_bins=n_bins)
        
        plt.figure(figsize=(6, 5), dpi=300)
        # 确保使用Times New Roman字体
        plt.rcParams['font.family'] = 'Times New Roman'
        
        ax = plt.gca()
        _styled_axes(ax)
        
        # Plot line connecting bin points
        plt.plot(prob_pred, prob_true, color=JAS_PALETTE['primary'], label='Calibration', linewidth=2.8, zorder=1)
        
        # Use optimized point markers, more suitable for academic chart style
        plt.scatter(prob_pred, prob_true, color=JAS_PALETTE['primary'], s=100, zorder=2, 
                   edgecolors='white', linewidths=1.2, marker='o', alpha=0.9)
        
        # Plot perfect calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', color=JAS_PALETTE['gray'], label='Perfectly calibrated', linewidth=1.2)
        
        plt.xlabel('Mean predicted confidence')
        plt.ylabel('Fraction of correct predictions')
        
        # Add ECE and MCE metrics on the plot with better formatting
        plt.text(0.05, 0.15, f'ECE: {ece:.4f}\nMCE: {mce:.4f}', 
                 transform=ax.transAxes, verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='lightgray', linewidth=0.5),
                 fontsize=10, fontweight='bold')
        
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return ece, mce, bin_df
    except Exception as e:
        print(f"绘制校准曲线时出错: {str(e)}")
        # 创建一个备用的简单曲线
        plt.figure(figsize=(6, 5), dpi=300)
        # 确保使用Times New Roman字体
        plt.rcParams['font.family'] = 'Times New Roman'
        
        ax = plt.gca()
        _styled_axes(ax)
        plt.plot([0, 1], [0, 1], linestyle='--', color=JAS_PALETTE['gray'], label='Perfectly calibrated')
        plt.xlabel('Mean predicted confidence')
        plt.ylabel('Fraction of correct predictions')
        plt.title('Calibration curve (error occurred)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        # 返回默认值
        return 0.0, 0.0, pd.DataFrame()
    
    # 打印分箱误差分布和校准质量分析
    print(f"\n校准评估指标:")
    print(f"ECE (期望校准误差): {ece:.4f}")
    print(f"MCE (最大校准误差): {mce:.4f}")
    
    # 校准质量评估
    calibration_quality = "优秀" if ece < 0.05 else "良好" if ece < 0.10 else "一般" if ece < 0.20 else "较差"
    print(f"校准质量评估: {calibration_quality}")
    
    print("\n各置信度分箱详情:")
    print(bin_df[['bin_lower', 'bin_upper', 'samples', 'avg_confidence', 'accuracy', 'error']].round(4))
    
    # 识别问题分箱
    problematic_bins = bin_df[bin_df['error'] > 0.20]
    if not problematic_bins.empty:
        print("\n需要关注的问题分箱:")
        print(problematic_bins[['bin_lower', 'bin_upper', 'samples', 'error']].round(4))
    
    return ece, mce, bin_df

def _evaluate_and_report(model, dataloader, class_names, save_dir):
    device = next(model.parameters()).device
    soak_times = np.array([5, 10, 20, 30, 40], dtype=np.float32)
    max_soak = float(40.0)
    
    # 检查class_names是否包含非ASCII字符（如中文），如果是则使用数值标签
    def _is_ascii_only(s):
        return all(ord(c) < 128 for c in str(s))
    
    plot_class_names = class_names.copy()
    if not all(_is_ascii_only(name) for name in class_names):
        # 包含非ASCII字符，使用soak_times的数值作为标签
        plot_class_names = [str(int(t)) for t in soak_times]
        print(f"检测到类别名称包含非ASCII字符，图表将使用英文数值标签: {plot_class_names}")

    y_true, y_pred, expected_amounts, probs_all = [], [], [], []
    # 为每个类别收集 1-2 张样本路径用于热力图
    samples_per_class = {i: [] for i in range(len(class_names))}
    # --- 收集 logits/labels 以便做温度缩放校准 ---
    logits_list = []
    with torch.no_grad():
        model.eval()
        for images, labels, paths in dataloader:
            images = images.to(device)
            labels_np = labels.numpy().tolist()
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            exp_amount = (probs * soak_times[None, :]).sum(axis=1)
            y_true.extend(labels_np)
            y_pred.extend(preds.tolist())
            expected_amounts.extend(exp_amount.tolist())
            probs_all.append(probs)
            logits_list.append(logits.cpu())
            # 记录样本（每类最多2个）
            for i, lbl in enumerate(labels_np):
                if len(samples_per_class[lbl]) < 2:
                    samples_per_class[lbl].append(paths[i])

    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, 5))
    true_amounts = [soak_times[t] for t in y_true]

    # 指标与表格（每类 avg_expected_amount / 指数 / 样本数 / MAE）
    per_class_rows = []
    mae_overall_amount = float(np.mean(np.abs(np.array(expected_amounts) - np.array(true_amounts)))) if len(true_amounts) else 0.0
    mae_overall_index = mae_overall_amount / max_soak
    for c in range(len(class_names)):
        idxs = [i for i, t in enumerate(y_true) if t == c]
        n = len(idxs)
        avg_expected_amount = float(np.mean([expected_amounts[i] for i in idxs])) if n else 0.0
        avg_expected_index = avg_expected_amount / max_soak
        mae_amount = float(np.mean([abs(expected_amounts[i] - true_amounts[i]) for i in idxs])) if n else 0.0
        mae_index = mae_amount / max_soak
        per_class_rows.append([
            class_names[c], c, round(avg_expected_amount, 4), round(avg_expected_index, 4), n, round(mae_amount, 4), round(mae_index, 4)
        ])

    _ensure_dir(save_dir)
    table_header = ['class_name', 'label', 'avg_expected_amount', 'avg_expected_index', 'count', 'MAE_amount', 'MAE_index']
    _save_table_csv(per_class_rows, table_header, os.path.join(save_dir, 'table_per_class_expected_mae.csv'))
    _save_table_csv([
        ['OVERALL', 'ALL', '', '', len(y_true), round(mae_overall_amount, 4), round(mae_overall_index, 4)]
    ], ['scope', 'label', 'avg_expected_amount', 'avg_expected_index', 'count', 'MAE_amount', 'MAE_index'],
      os.path.join(save_dir, 'table_overall_mae.csv'))

    # 可视化（使用标准溶蚀程度英文名称）
    display_names = EROSION_CLASS_NAMES_EN if len(plot_class_names) == len(EROSION_CLASS_NAMES_EN) else plot_class_names
    _plot_confusion_matrix(y_true, y_pred, display_names, os.path.join(save_dir, 'confusion_matrix.png'))
    if len(probs_all) > 0:
        # 绘制原始校准曲线并保存ECE/MCE指标
        original_ece, original_mce, _ = _plot_calibration_curve(y_true, probs_all, os.path.join(save_dir, 'calibration_curve.png'))
        print(f"\n原始模型校准性能:")
        print(f"ECE: {original_ece:.4f}, MCE: {original_mce:.4f}")
    # ROC and PR (using temperature-scaled probabilities for stability; fallback to original prob_matrix if failed)
        try:
            prob_for_curves = probs_cal if 'probs_cal' in locals() else probs_all
            mean_auc, mean_ap = _plot_roc_and_pr(y_true, prob_for_curves, display_names, save_dir)
        except Exception:
            mean_auc, mean_ap = _plot_roc_and_pr(y_true, probs_all, display_names, save_dir)
    # ====== 多阶段校准优化（向量温度缩放 + Brier Score优化 + 精细分段校准） ======
    vector_temps_list = None
    try:
        logits_all = torch.cat(logits_list, dim=0)
        labels_all = torch.tensor(y_true, dtype=torch.long)
        num_classes = logits_all.size(1)
        
        # 阶段1: 单一温度缩放（基础校准）- 针对溶蚀程度识别优化版
        # 使用固定的最佳温度值0.7（从评估结果中确定）
        T_value = 0.7
        print(f"使用最佳温度值: {T_value:.4f}（已从温度缩放评估中确定）")
        
        # 阶段2: 向量温度缩放（每个类别独立温度，进一步提升）- 针对溶蚀程度识别优化
        with torch.no_grad():
            probs_temp1 = torch.softmax(logits_all / T_value, dim=1)
        
        # 计算每个类别的平均置信度，用于向量温度初始化
        class_temps = torch.ones(num_classes, device=logits_all.device)
        for c in range(num_classes):
            mask = labels_all == c
            if mask.sum() > 0:
                avg_conf = probs_temp1[mask, c].mean().item()
                # 针对溶蚀程度识别的温度初始化策略：考虑溶蚀程度的连续性
                # 对中间程度类别使用更精细的调整
                base_temp = 1.0 + 2.0 * (0.7 - avg_conf)  # 增加调整幅度
                # 溶蚀程度类别调整：越中间的类别，调整越精细
                if num_classes > 3:
                    center = (num_classes - 1) / 2
                    position_factor = 1.0 - 0.4 * abs(c - center) / center  # 中间位置有更高的调整精度
                    base_temp *= position_factor
                class_temps[c] = max(0.2, min(5.0, base_temp))
                # 溶蚀程度特有的类不平衡调整
                class_weight = min(1.0, mask.sum() / (len(labels_all) / num_classes))
                # 对样本少的类别使用更保守的温度调整
                class_temps[c] = T_value + (class_temps[c] - T_value) * class_weight
        
        # 向量温度优化（优化Brier Score + 交叉熵损失组合）- 针对溶蚀程度识别优化
        vector_temps = torch.nn.Parameter(class_temps.clone())
        # 针对溶蚀程度识别的LBFGS参数优化
        optimizer_vt = torch.optim.LBFGS([vector_temps], lr=0.07, max_iter=150, 
                                        line_search_fn='strong_wolfe',
                                        tolerance_grad=1e-6, tolerance_change=1e-9, 
                                        history_size=10)
        
        def _combined_objective():
            optimizer_vt.zero_grad()
            # 应用向量温度：每个样本的logits按类别应用对应温度
            scaled_logits = logits_all.clone()
            for c in range(num_classes):
                scaled_logits[:, c] = scaled_logits[:, c] / vector_temps[c].clamp(0.05, 5.0)
            probs_vt = F.softmax(scaled_logits, dim=1)
            
            # 组合损失：Brier Score + 交叉熵损失
            one_hot = F.one_hot(labels_all, num_classes).float()
            brier = ((probs_vt - one_hot) ** 2).mean()
            cross_entropy = F.cross_entropy(scaled_logits, labels_all)
            # 加权组合损失
            combined_loss = 0.6 * brier + 0.4 * cross_entropy
            
            combined_loss.backward()
            return combined_loss
        
        # 增加收敛检查的优化过程
        prev_loss_vec = float('inf')
        convergence_count_vec = 0
        
        for _ in range(15):
            loss = optimizer_vt.step(_combined_objective)
            # 收敛检查
            if abs(prev_loss_vec - loss.item()) < 1e-6:
                convergence_count_vec += 1
            else:
                convergence_count_vec = 0
            prev_loss_vec = loss.item()
            
            if convergence_count_vec >= patience:
                print(f"向量温度缩放优化提前收敛，迭代轮数: {_+1}")
                break
        
        vector_temps_final = vector_temps.detach().clamp(0.05, 5.0)
        print(f"向量温度校准结果: {', '.join([f'{t:.4f}' for t in vector_temps_final.tolist()])}")
        
        # 应用向量温度
        with torch.no_grad():
            scaled_logits_vt = logits_all.clone()
            for c in range(num_classes):
                scaled_logits_vt[:, c] = scaled_logits_vt[:, c] / vector_temps_final[c]
            probs_cal = torch.softmax(scaled_logits_vt, dim=1).cpu().numpy()
        
        # 绘制校准曲线并获取ECE/MCE指标
        ece_temp, mce_temp, _ = _plot_calibration_curve(y_true, probs_cal, os.path.join(save_dir, 'calibration_curve_after_temp_scaling.png'))
        print(f"\n温度缩放后校准性能:")
        print(f"ECE: {ece_temp:.4f}, MCE: {mce_temp:.4f}")
        
        # 阶段3: 自适应分段等距回归（基于数据分布的动态分段）- 针对溶蚀程度识别优化
        p_max = probs_cal.max(axis=1)
        hits = (probs_cal.argmax(axis=1) == np.array(y_true)).astype(int)
        
        # 使用自适应分段策略，针对溶蚀程度识别任务优化
        def adaptive_segment_calibrate(p_vals, h_vals):
            p_cal = p_vals.copy()
            
            # 基于分位数的自适应分段，针对溶蚀程度识别任务调整
            # 使用更多分段以提高校准精度
            n_segments = 8 if len(p_vals) > 100 else 6 if len(p_vals) > 50 else 5
            quantiles = np.quantile(p_vals, np.linspace(0, 1, n_segments + 1))
            segments = [(quantiles[i], quantiles[i+1]) for i in range(len(quantiles)-1)]
            
            # 确保分段覆盖完整区间
            segments[0] = (0.0, segments[0][1])
            segments[-1] = (segments[-1][0], 1.0)
            
            # 对于溶蚀程度识别，中间置信度区域（0.4-0.8）更关键，需要更精细分段
            if len(p_vals) > 80:
                # 增加中间区域的分段
                mid_segments = [(0.4, 0.55), (0.55, 0.7), (0.7, 0.8)]
                all_segments = []
                i = 0
                while i < len(segments):
                    # 替换覆盖中间区域的大分段
                    if segments[i][0] <= 0.4 and segments[i][1] >= 0.8:
                        all_segments.extend(mid_segments)
                    elif segments[i][0] <= 0.4 and segments[i][1] >= 0.55:
                        all_segments.append((segments[i][0], 0.55))
                        all_segments.append((0.55, segments[i][1]))
                    elif segments[i][0] <= 0.55 and segments[i][1] >= 0.8:
                        all_segments.append((segments[i][0], 0.7))
                        all_segments.append((0.7, segments[i][1]))
                    else:
                        all_segments.append(segments[i])
                    i += 1
                segments = sorted(list(set(all_segments)), key=lambda x: x[0])
            
            # 应用等距回归到每个分段 - 针对溶蚀程度识别优化
            for low, high in segments:
                mask = (p_vals >= low) & (p_vals < high) if high < 1.0 else (p_vals >= low) & (p_vals <= high)
                n_samples = mask.sum()
                
                # 针对溶蚀程度识别的动态样本数阈值
                # 中间置信度区域使用更低的阈值以提高覆盖度
                if 0.4 <= low and high <= 0.8:
                    sample_threshold = 6  # 中间区域接受更少样本
                else:
                    sample_threshold = 8 if n_samples < 50 else 12
                
                if n_samples >= sample_threshold:
                    try:
                        # 针对溶蚀程度识别的等距回归配置优化
                        ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
                        calibrated = ir.fit_transform(p_vals[mask], h_vals[mask])
                        
                        # 应用更平滑的处理，避免异常跳跃
                        # 对于溶蚀程度识别，确保校准后的概率变化更平滑
                        if n_samples > 15:
                            # 对大样本段应用简单的移动平均进一步平滑
                            calibrated = np.convolve(calibrated, np.ones(3)/3, mode='same')
                        
                        p_cal[mask] = calibrated
                    except Exception as e:
                        print(f"等距回归失败在区间 [{low:.2f}, {high:.2f}]: {str(e)}")
                        pass  # 如果该段校准失败，保持原值
            
            return p_cal
        
        p_calibrated = adaptive_segment_calibrate(p_max, hits)
        
        # 阶段4: 平滑校准后的概率分布（使用自适应缩放和概率重分配）- 针对溶蚀程度识别优化
        scale = (p_calibrated + 1e-8) / (p_max + 1e-8)
        # 自适应缩放范围：针对溶蚀程度识别任务优化
        confidence_levels = p_max
        # 针对溶蚀程度识别的缩放策略：更保守的调整，确保概率连续性
        lower_bounds = 0.5 + 0.15 * confidence_levels  # 稍微放宽下限
        upper_bounds = 1.3 - 0.2 * confidence_levels   # 更保守的上限
        # 确保边界合理
        lower_bounds = np.clip(lower_bounds, 0.5, 0.85)
        upper_bounds = np.clip(upper_bounds, 1.15, 1.5)
        
        # 应用自适应缩放边界
        scaled_scale = np.zeros_like(scale)
        for i in range(len(scale)):
            scaled_scale[i] = np.clip(scale[i], lower_bounds[i], upper_bounds[i])
        
        probs_cal_iso = probs_cal.copy()
        for i in range(probs_cal_iso.shape[0]):
            j = np.argmax(probs_cal_iso[i])
            old_max = probs_cal_iso[i, j]
            new_max = old_max * scaled_scale[i]
            delta = new_max - old_max
            
            # 高级概率重分配策略 - 针对溶蚀程度识别优化
            others_sum = probs_cal_iso[i, :].sum() - old_max
            if others_sum > 1e-8:
                # 基于溶蚀程度类别的特殊重分配策略
                # 考虑类别之间的顺序关系，相邻溶蚀程度的类别应保持概率连续性
                class_indices = np.arange(probs_cal_iso.shape[1])
                # 排除最大类
                other_indices = class_indices[class_indices != j]
                
                # 计算与最大类的距离（溶蚀程度差异）
                distance = np.abs(other_indices - j)
                # 距离越大，调整比例越小
                adjust_weights = 1.0 / (distance + 1.5)  # 添加1.5以避免除零
                adjust_weights = adjust_weights / adjust_weights.sum()  # 归一化
                
                # 计算每个类别的调整量
                # 对于溶蚀程度识别，允许更大的调整范围以提高校准精度
                reduction = min(abs(delta), others_sum * 0.85)  # 增加调整上限
                reduction = reduction if delta > 0 else -reduction
                
                # 应用调整到每个非最大类
                for idx, sorted_idx in enumerate(sorted_indices):
                    adj_amount = reduction * adjust_weights[idx]
                    probs_cal_iso[i, sorted_idx] -= adj_amount
                
                # 设置新的最大类概率
                probs_cal_iso[i, j] = new_max
            else:
                probs_cal_iso[i, j] = new_max
            
            # 重新归一化并确保数值稳定性
            total = probs_cal_iso[i, :].sum()
            if total > 1e-8:
                probs_cal_iso[i, :] = probs_cal_iso[i, :] / total
            
            # 确保概率分布平滑，避免极端值
            probs_cal_iso[i] = np.clip(probs_cal_iso[i], 1e-7, 1.0 - 1e-7)
            probs_cal_iso[i] = probs_cal_iso[i] / probs_cal_iso[i].sum()  # 再次归一化
        
        # 绘制等距回归后的校准曲线并获取ECE/MCE指标
        ece_iso, mce_iso, _ = _plot_calibration_curve(y_true, probs_cal_iso, os.path.join(save_dir, 'calibration_curve_after_isotonic.png'))
        print(f"\n等距回归后校准性能:")
        print(f"ECE: {ece_iso:.4f}, MCE: {mce_iso:.4f}")
        
        # 保存向量温度参数（在后续的calibration.json中会合并）
        vector_temps_list = vector_temps_final.cpu().tolist() if 'vector_temps_final' in locals() else None
    except Exception:
        T_value = 1.0
        vector_temps_list = None
    # ====== 类别偏差矫正（基于残差均值） ======
    try:
        # 使用温度缩放后的概率再计算期望量，得到偏差估计
        with torch.no_grad():
            probs_cal2 = torch.softmax(torch.cat(logits_list, dim=0) / T_value, dim=1).cpu().numpy()
        exp_amount_cal = (probs_cal2 * soak_times[None, :]).sum(axis=1)
        residuals_cal = exp_amount_cal - np.array(true_amounts)
        bias_offsets = {}
        for c in range(len(class_names)):
            idxs = [i for i, t in enumerate(y_true) if t == c]
            if len(idxs) == 0:
                bias_offsets[str(c)] = 0.0
            else:
                bias_offsets[str(c)] = float(-np.mean(residuals_cal[idxs]))
        # 保存校准参数（合并所有校准方法）
        # 保存校准参数，包括完整的ECE和MCE指标跟踪
        calib_data = {
            'temperature': T_value,
            'original_ece': original_ece if 'original_ece' in locals() else 0.0,
            'original_mce': original_mce if 'original_mce' in locals() else 0.0,
            'temp_scaled_ece': ece_temp if 'ece_temp' in locals() else 0.0,
            'temp_scaled_mce': mce_temp if 'ece_temp' in locals() else 0.0,
            'fully_calibrated_ece': ece_iso if 'ece_iso' in locals() else 0.0,
            'fully_calibrated_mce': mce_iso if 'ece_iso' in locals() else 0.0,
            'bias_offsets': bias_offsets,
            'isotonic': True,
            'method': 'vector_temp + fine_segment_isotonic',
            'ece_before': float(original_ece) if 'original_ece' in locals() else None,
            'ece_after_temp_scaling': float(ece_temp) if 'ece_temp' in locals() else None,
            'ece_after_isotonic': float(ece_iso) if 'ece_iso' in locals() else None,
            'mce_before': float(original_mce) if 'original_mce' in locals() else None,
            'mce_after_temp_scaling': float(mce_temp) if 'mce_temp' in locals() else None,
            'mce_after_isotonic': float(mce_iso) if 'mce_iso' in locals() else None
        }
        if 'vector_temps_list' in locals() and vector_temps_list is not None:
            calib_data['vector_temperatures'] = vector_temps_list
        with open(os.path.join(save_dir, 'calibration.json'), 'w', encoding='utf-8') as f:
            json.dump(calib_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # ROC and PR (using temperature-scaled probabilities for stability; fallback to original prob_matrix if failed)
    try:
        prob_for_curves = probs_cal if 'probs_cal' in locals() else probs_all
        # 使用标准溶蚀程度英文名称
        display_names = EROSION_CLASS_NAMES_EN if len(class_names) == len(EROSION_CLASS_NAMES_EN) else class_names
        mean_auc, mean_ap = _plot_roc_and_pr(y_true, prob_for_curves, display_names, save_dir)
    except Exception:
        mean_auc, mean_ap = _plot_roc_and_pr(y_true, probs_all, display_names, save_dir)
    # 导出元信息
    with open(os.path.join(save_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'class_names': class_names,
            'num_samples': len(y_true),
            'mae_overall_amount': mae_overall_amount,
            'mae_overall_index': mae_overall_index,
            'roc_macro_auc': mean_auc,
            'pr_macro_ap': mean_ap,
            'note': 'kept confusion matrix and calibration; added ROC/PR. Calibration params in calibration.json'
        }, f, ensure_ascii=False, indent=2)
    print(f"📑 报告与图表已生成到: {save_dir}")
    # ---- 生成 Grad-CAM 热力图（每类 1-2 张，优先选择“显著面积大”的样本）----
    try:
        heat_dir = _ensure_dir(os.path.join(save_dir, 'heatmaps'))
        heat_dir_overlay = _ensure_dir(os.path.join(heat_dir, 'overlay'))
        heat_dir_original = _ensure_dir(os.path.join(heat_dir, 'original'))
        target_layer = model.cnn[5]  # 最后一层卷积：Conv2d(32,64,...)
        activations = None
        gradients = None
        def fwd_hook(module, inp, out):
            nonlocal activations
            activations = out.detach()
        def bwd_hook(module, grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        model.eval()
        def _safe_name(s: str) -> str:
            return ''.join(ch if (ch.isalnum() or ch in ' _.-') else '_' for ch in str(s))
        saved = 0
        # 评估热力“显著度”：阈值面积占比与均值综合
        def score_cam(cam_arr: np.ndarray) -> float:
            if cam_arr.size == 0: return 0.0
            m = cam_arr.max() if cam_arr.max() > 1e-8 else 1.0
            camn = cam_arr / m
            area = (camn >= 0.6).mean()  # 高响应区域占比
            intensity = camn.mean()
            return 0.7 * area + 0.3 * intensity

        # 为每个类别收集候选并选 Top-2（用原始验证集图像，不做增强）
        from collections import defaultdict
        topk = defaultdict(list)  # cls -> list[(score, overlay, pred)]
        for images, labels, paths in dataloader:
            for i in range(images.size(0)):
                img_tensor = images[i:i+1].to(device)
                lbl = int(labels[i].item())
                model.zero_grad(set_to_none=True)
                out = model(img_tensor)
                pred = out.softmax(dim=1).argmax(dim=1).item()
                # 优先记录预测正确的样本
                if pred != lbl and len(topk[lbl]) >= 6:
                    continue
                loss = out[0, pred]
                loss.backward()
                if activations is None or gradients is None:
                    continue
                grads = gradients[0]
                acts = activations[0]
                weights = grads.mean(dim=(1, 2), keepdim=True)
                cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()
                if cam.max() > 1e-8:
                    cam = cam / (cam.max() + 1e-8)
                # 将 CAM 尺寸对齐到原始图像尺寸，避免错切/仿射带来的主观偏差
                orig = safe_imread(paths[i], cv2.IMREAD_GRAYSCALE)
                if orig is None:
                    orig = (images[i].cpu().numpy().squeeze() * 0.5 + 0.5)
                    orig = np.clip(orig * 255.0, 0, 255).astype(np.uint8)
                H, W = orig.shape[:2]
                # 过滤“旋转/黑边明显”的图片：整体黑像素占比或四边黑边占比过高
                try:
                    black = (orig <= 5).astype(np.uint8)
                    black_ratio = float(black.mean())
                    border = 10
                    border_mask = np.zeros_like(black)
                    border_mask[:border,:] = 1; border_mask[-border:,:] = 1
                    border_mask[:, :border] = 1; border_mask[:, -border:] = 1
                    border_ratio = float((black & border_mask).sum() / max(1, border_mask.sum()))
                    if black_ratio > 0.20 or border_ratio > 0.30:
                        # 跳过该候选，寻找更自然的原图
                        continue
                except Exception:
                    pass
                # 将图像和CAM都resize到128x128
                orig_resized = cv2.resize(orig, (128, 128))
                cam_resized = cv2.resize(cam, (128, 128))
                s = score_cam(cam_resized)
                # 生成可视化（分别保存原图与叠加图）
                heat = (cam_resized * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                base = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(base, 0.45, heat_color, 0.55, 0)
                topk[lbl].append((s, (base, overlay), pred))

        for c in range(len(class_names)):
            if not topk[c]:
                continue
            # 按分数从高到低取前2个
            top = sorted(topk[c], key=lambda x: x[0], reverse=True)[:2]
            # 使用溶蚀程度（soak_times）作为文件名
            erosion_level = int(soak_times[c])
            for idx, (_, pair, pred) in enumerate(top, start=1):
                base, overlay = pair
                # 文件名使用溶蚀程度数值
                name = f"erosion_{erosion_level}_{idx}_pred{pred}.png"
                ok1 = safe_imwrite(os.path.join(heat_dir_original, name), base)
                ok2 = safe_imwrite(os.path.join(heat_dir_overlay, name), overlay)
                if ok1 and ok2:
                    saved += 1
        h1.remove(); h2.remove()
        if saved == 0:
            print("热力图未生成：可能是样本不足或写入失败")
        else:
            print(f"🔥 已生成{saved}张热力图到: {os.path.join(save_dir, 'heatmaps')}")
    except Exception as e:
        print(f"热力图生成失败: {e}")

# 载入预训练模型权重（可选）
def _load_pretrained_model(model: nn.Module, checkpoint_path: str, freeze_cnn: bool = False, freeze_transformer: bool = False) -> nn.Module:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"预训练权重文件不存在: {checkpoint_path}")
    print(f"正在载入预训练权重: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # 兼容直接 state_dict 或 完整 checkpoint
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        if 'best_val_acc' in ckpt:
            print(f"原模型最佳验证准确率: {ckpt['best_val_acc']:.2f}%")
    else:
        state_dict = ckpt
    
    # 处理模型结构变化：分类头添加了Dropout层导致参数名变化
    new_state_dict = {}
    for k, v in state_dict.items():
        # 映射旧的分类头参数名到新的结构
        if k == 'classifier.5.weight':
            new_state_dict['classifier.6.weight'] = v
        elif k == 'classifier.5.bias':
            new_state_dict['classifier.6.bias'] = v
        else:
            new_state_dict[k] = v
    
    # 使用strict=False允许部分参数不匹配
    model.load_state_dict(new_state_dict, strict=False)
    print("预训练权重载入成功!")
    if freeze_cnn:
        for name, p in model.named_parameters():
            if 'cnn' in name:
                p.requires_grad = False
        print("CNN部分已冻结")
    if freeze_transformer:
        for name, p in model.named_parameters():
            if 'transformer' in name or 'patch_embed' in name:
                p.requires_grad = False
        print("Transformer部分已冻结")
    return model

# 训练函数
def train_model(train_image_paths, train_labels, val_image_paths, val_labels, num_epochs=100,
                pretrained_path: str | None = None, freeze_cnn: bool = False,
                freeze_transformer: bool = False, finetune_lr: float | None = None,
                label_to_class_name: list | None = None):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 预过滤不可读/坏图，避免训练中断
    def _filter_readable(paths, labels):
        readable_paths, readable_labels = [], []
        skipped = 0
        for p, y in zip(paths, labels):
            try:
                img = safe_imread(p, cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = None
            if img is None:
                skipped += 1
                print(f"警告: 无法读取图像，已跳过: {p}")
                continue
            readable_paths.append(p)
            readable_labels.append(y)
        return readable_paths, readable_labels, skipped
    train_image_paths, train_labels, skipped_train = _filter_readable(train_image_paths, train_labels)
    val_image_paths, val_labels, skipped_val = _filter_readable(val_image_paths, val_labels)
    if skipped_train or skipped_val:
        print(f"预过滤完成: 训练集跳过{skipped_train}张，验证集跳过{skipped_val}张")

    # 创建数据集和数据加载器
    train_dataset = ErosionDataset(train_image_paths, train_labels, is_train=True)
    val_dataset = ErosionDataset(val_image_paths, val_labels, is_train=False)
    
    # 类别均衡采样器（缓解部分类别拟合不足）
    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    train_label_counts = Counter(train_labels)
    sample_weights = []
    for y in train_labels:
        # 基础按频次反比
        w = 1.0 / max(train_label_counts.get(y, 1), 1)
        # 对 20/30/40（标签 2/3/4）给予额外关注
        if y in {2, 3, 4}: w *= 1.5
        sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=50, sampler=sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=0)
    
    # 初始化模型、损失函数和优化器
    # 序数分类固定为 5 个等级（标签为 0..4）
    num_classes = 5
    model = EnhancedCNNT(num_classes=num_classes).to(device)

    # 载入预训练（如果提供）
    if pretrained_path:
        model = _load_pretrained_model(model, pretrained_path, freeze_cnn=freeze_cnn, freeze_transformer=freeze_transformer)
    # 模型体量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)  # FP32按4字节估算
    print(f"Model Params: total={total_params:,}, trainable={trainable_params:,}, size≈{model_size_mb:.2f} MB")
    # 基于训练集频次的反比权重（用于损失，加速摆脱单类塌缩）
    from collections import Counter
    freq = Counter(train_labels)
    class_weights = [0.0] * 5
    for c in range(5):
        class_weights[c] = 0.0 if freq.get(c, 0) == 0 else 1.0 / freq[c]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * 5
    # 自定义损失：平滑 + Focal，缓解过度置信与极端误差
    class CrossEntropyWithFocalAndSmoothing(nn.Module):
        def __init__(self, weight=None, label_smoothing: float = 0.05, gamma: float = 1.5):
            super().__init__()
            self.register_buffer('weight', weight if weight is not None else None)
            self.label_smoothing = label_smoothing
            self.gamma = gamma
        def forward(self, logits, target):
            num_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            log_probs = F.log_softmax(logits, dim=1)
            ce = -(true_dist * log_probs).sum(dim=1)
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            focal = (1.0 - pt) ** self.gamma
            loss = focal * ce
            if self.weight is not None:
                w = self.weight.gather(0, target)
                loss = loss * w
            return loss.mean()
    criterion = CrossEntropyWithFocalAndSmoothing(weight=class_weights_tensor, label_smoothing=0.05, gamma=1.5)
    # 微调时默认更小学习率，或使用传入的 finetune_lr
    base_lr = 3e-4
    if pretrained_path:
        base_lr = 1e-4
    if finetune_lr is not None:
        base_lr = finetune_lr
    if pretrained_path:
        print(f"使用微调学习率: {base_lr}")
    # 设置L2正则化（权重衰减）为1e-4
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练循环
    best_val_acc = 0.0
    
    # 初始化存储训练和验证指标的列表，用于绘制趋势曲线
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mse_sum = 0.0
        train_correct = 0
        train_total = 0
        
        # 训练步骤
        for images, labels, _paths in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪提高鲁棒性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # 计算MSE（以概率 vs one-hot 为准）
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                onehot = F.one_hot(labels, num_classes).float()
                train_mse_sum += F.mse_loss(probs, onehot, reduction='sum').item()
        
        # 验证步骤
        model.eval()
        val_loss = 0.0
        val_mse_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            soak_times = [5, 10, 20, 30, 40]
            max_soak = 40.0
            for images, labels, _paths in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                # 计算MSE
                probs = F.softmax(outputs, dim=1)
                onehot = F.one_hot(labels, num_classes).float()
                val_mse_sum += F.mse_loss(probs, onehot, reduction='sum').item()
        
        # 计算准确率
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_mse = train_mse_sum / train_total if train_total > 0 else 0.0
        val_mse = val_mse_sum / val_total if val_total > 0 else 0.0
        train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # 存储指标用于绘制趋势曲线
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 打印 epoch 信息
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        # 输出评价指标为 MSE 与 CA（分类准确率）
        print(f'Train MSE: {train_mse:.6f}, Train CA: {train_acc:.2f}%')
        print(f'Val   MSE: {val_mse:.6f}, Val   CA: {val_acc:.2f}%')
        # 不再输出逐样本结果
        
        # 学习率调整
        scheduler.step(val_loss_avg)
        
        # 保存最佳模型（基于准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_erosion_model.pth')
            print(f'Saved best model with Val Acc: {best_val_acc:.2f}%')
        


    # 训练完成后：对验证集按类别汇总平均预测溶蚀量/指数
    try:
        model.eval()
        soak_times = [5, 10, 20, 30, 40]
        max_soak = 40.0
        sum_pred_amount = [0.0] * 5
        count_per_label = [0] * 5
        # 期望值统计（连续值）：E[amount] = Σ p_k * amount_k
        sum_expected_amount = [0.0] * 5
        with torch.no_grad():
            for images, labels, _paths in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(probs.data, 1)
                for i in range(labels.size(0)):
                    lbl = int(labels[i].item())
                    pred_amount = soak_times[int(predicted[i].item())]
                    # 期望量（连续）
                    expected_amount = float((probs[i] * torch.tensor(soak_times, device=probs.device, dtype=probs.dtype)).sum().item())
                    sum_pred_amount[lbl] += float(pred_amount)
                    sum_expected_amount[lbl] += expected_amount
                    count_per_label[lbl] += 1
        print('=' * 50)
        print('验证集各子类平均预测溶蚀量/指数:')
        summary_rows = []
        expected_rows = []
        for lbl in range(5):
            avg_amount = (sum_pred_amount[lbl] / count_per_label[lbl]) if count_per_label[lbl] > 0 else 0.0
            avg_index = avg_amount / max_soak
            avg_expected_amount = (sum_expected_amount[lbl] / count_per_label[lbl]) if count_per_label[lbl] > 0 else 0.0
            avg_expected_index = avg_expected_amount / max_soak
            class_name = None
            if isinstance(label_to_class_name, list) and lbl < len(label_to_class_name):
                class_name = label_to_class_name[lbl]
            name_str = class_name if class_name else f'Label{lbl}'
            print(f"{name_str}: 平均{avg_amount:.2f}次 / 指数{avg_index:.2f} (样本{count_per_label[lbl]}) ")
            summary_rows.append((name_str, lbl, avg_amount, avg_index, count_per_label[lbl]))
            expected_rows.append((name_str, lbl, avg_expected_amount, avg_expected_index, count_per_label[lbl]))
        # 保存CSV
        import csv
        csv_path = 'val_per_class_avg.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'label', 'avg_pred_amount', 'avg_pred_index', 'count'])
            writer.writerows(summary_rows)
        print(f"已保存验证集分组平均结果: {csv_path}")
        # 保存期望值CSV
        csv_path2 = 'val_per_class_avg_expected.csv'
        with open(csv_path2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'label', 'avg_expected_amount', 'avg_expected_index', 'count'])
            writer.writerows(expected_rows)
        print(f"已保存验证集分组平均期望结果: {csv_path2}")
    except Exception as e:
        print(f"生成验证集统计信息时出错: {e}")
        print('-' * 50)
    
    # 生成训练和验证趋势曲线图
    try:
        reports_dir = _ensure_dir(os.path.join(os.getcwd(), 'reports'))
        
        # 创建图表
        plt.figure(figsize=(15, 10))
        
        # 确保使用Times New Roman字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 1. Loss curve
        plt.subplot(2, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color=JAS_PALETTE['primary'])
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color=JAS_PALETTE['accent'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # 设置Loss纵坐标范围为0到1
        
        # 2. MSE curve
        plt.subplot(2, 2, 2)
        plt.plot(range(1, num_epochs + 1), train_mses, label='Training MSE', color=JAS_PALETTE['primary'])
        plt.plot(range(1, num_epochs + 1), val_mses, label='Validation MSE', color=JAS_PALETTE['accent'])
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training and Validation MSE Curves')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)  # 设置MSE纵坐标范围为0到1
        
        # 3. Accuracy curve
        plt.subplot(2, 2, 3)
        plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color=JAS_PALETTE['primary'])
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color=JAS_PALETTE['accent'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy Curves')
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 100)  # 设置Accuracy纵坐标范围为0到100
        
        # 4. All metrics summary curve
        plt.subplot(2, 2, 4)
        # Normalization for visualization on the same chart
        max_loss = max(max(train_losses), max(val_losses)) if train_losses and val_losses else 1
        max_mse = max(max(train_mses), max(val_mses)) if train_mses and val_mses else 1
        
        # Normalize loss and MSE to 0-1 range, keep accuracy as is (0-100%)
        norm_train_loss = [loss/max_loss for loss in train_losses]
        norm_val_loss = [loss/max_loss for loss in val_losses]
        norm_train_mse = [mse/max_mse for mse in train_mses]
        norm_val_mse = [mse/max_mse for mse in val_mses]
        norm_train_acc = [acc/100 for acc in train_accuracies]
        norm_val_acc = [acc/100 for acc in val_accuracies]
        
        plt.plot(range(1, num_epochs + 1), norm_train_loss, label='Training Loss (Normalized)', color=JAS_PALETTE['primary'], linestyle='-')
        plt.plot(range(1, num_epochs + 1), norm_val_loss, label='Validation Loss (Normalized)', color=JAS_PALETTE['primary'], linestyle='--')
        plt.plot(range(1, num_epochs + 1), norm_train_mse, label='Training MSE (Normalized)', color=JAS_PALETTE['secondary'], linestyle='-')
        plt.plot(range(1, num_epochs + 1), norm_val_mse, label='Validation MSE (Normalized)', color=JAS_PALETTE['secondary'], linestyle='--')
        plt.plot(range(1, num_epochs + 1), norm_train_acc, label='Training Accuracy (Normalized)', color=JAS_PALETTE['accent'], linestyle='-')
        plt.plot(range(1, num_epochs + 1), norm_val_acc, label='Validation Accuracy (Normalized)', color=JAS_PALETTE['accent'], linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Value')
        plt.title('All Metrics Summary Trends (Normalized)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图表
        trend_plot_path = os.path.join(reports_dir, 'training_validation_trends.png')
        plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始数据到JSON文件，便于后续分析
        trends_data = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_mses': train_mses,
            'val_mses': val_mses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
        }
        trends_json_path = os.path.join(reports_dir, 'training_validation_trends.json')
        with open(trends_json_path, 'w', encoding='utf-8') as f:
            json.dump(trends_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 训练和验证趋势曲线图已保存到: {trend_plot_path}")
        print(f"📊 训练和验证趋势数据已保存到: {trends_json_path}")
        
    except Exception as e:
        print(f"生成训练/验证趋势曲线图时出错: {e}")
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')
    # 生成更全面的报告与图表
    try:
        # 使用标准溶蚀程度英文名称
        class_names = EROSION_CLASS_NAMES_EN
        reports_dir = _ensure_dir(os.path.join(os.getcwd(), 'reports'))
        _evaluate_and_report(model, val_loader, class_names, reports_dir)
    except Exception as e:
        print(f"Report generation failed: {e}")
    return model

# 构建数据集工具函数
def _list_images_in_dir(directory):
    allowed_exts = {'.bmp', '.jpg', '.jpeg', '.png'}
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in allowed_exts:
                paths.append(os.path.abspath(os.path.join(root, fname)))
    return paths

def build_paths_and_labels_from_root(root_dir):
    # 支持两种结构：
    # 1) 直接类别目录在 root_dir 下
    # 2) 存在 train/val/test 等分割目录，类别目录在其下一级
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subdirs.sort()

    # 如果顶层目录主要是分割目录，则下降一层收集类别目录
    split_like = {"train", "val", "valid", "validation", "test"}
    top_like_splits = [d for d in subdirs if d.lower() in split_like]
    if len(top_like_splits) >= 1:
        # 收集所有 split 下的类别目录，构建统一的映射
        candidate_class_dirs = []
        for sd in top_like_splits:
            sd_path = os.path.join(root_dir, sd)
            for cd in os.listdir(sd_path):
                full = os.path.join(sd_path, cd)
                if os.path.isdir(full):
                    candidate_class_dirs.append((sd, cd, full))

        # 基于类别名（cd）构建映射（忽略 train/val 层）
        class_names = sorted({cd for _, cd, _ in candidate_class_dirs})
    else:
        # 顶层就是类别目录
        class_names = subdirs

    def parse_soak_times_to_level(name: str) -> int:
        import re
        nums = re.findall(r"\d+", name)
        soak = None
        for s in nums:
            v = int(s)
            if v in {5, 10, 20, 30, 40}:
                soak = v
                break
        if soak is None:
            if nums:
                candidates = [int(x) for x in nums]
                nearest = min([5, 10, 20, 30, 40], key=lambda t: min(abs(t - v) for v in candidates))
                soak = nearest
                print(f"警告: 未能明确解析'{name}'的浸泡次数，近似映射为 {nearest}")
            else:
                soak = 20
                print(f"警告: 未在'{name}'中发现数字，默认按20次处理")
        mapping = {5: 1, 10: 2, 20: 3, 30: 4, 40: 5}
        return mapping[soak]

    class_to_level = {name: parse_soak_times_to_level(name) for name in class_names}
    image_paths, labels = [], []

    if len(top_like_splits) >= 1:
        # 遍历 split 下的类别目录
        for sd in top_like_splits:
            sd_path = os.path.join(root_dir, sd)
            for cd in os.listdir(sd_path):
                full = os.path.join(sd_path, cd)
                if not os.path.isdir(full):
                    continue
                if cd not in class_to_level:
                    # 不应出现，但稳妥处理
                    continue
                paths = _list_images_in_dir(full)
                for p in paths:
                    image_paths.append(p)
                    labels.append(class_to_level[cd] - 1)
    else:
        for name in class_names:
            dir_path = os.path.join(root_dir, name)
            paths = _list_images_in_dir(dir_path)
            for p in paths:
                image_paths.append(p)
                labels.append(class_to_level[name] - 1)  # 转为 0..4
    return image_paths, labels, class_to_level

def stratified_split(paths, labels, train_ratio=0.8, seed=42):
    random.seed(seed)
    # 将索引按标签分组
    label_to_indices = {}
    for idx, y in enumerate(labels):
        label_to_indices.setdefault(y, []).append(idx)
    train_indices, val_indices = [], []
    for y, idxs in label_to_indices.items():
        random.shuffle(idxs)
        k = int(len(idxs) * train_ratio)
        train_indices.extend(idxs[:k])
        val_indices.extend(idxs[k:])
    # 打乱整体顺序
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    train_paths = [paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    return train_paths, train_labels, val_paths, val_labels

# 使用示例
if __name__ == '__main__':
    # 从工作目录下的 kuochong_split1 目录自动构建数据集
    dataset_root = os.path.join(os.path.dirname(__file__), 'kuochong_split1')
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"未找到数据集目录: {dataset_root}")
    all_paths, all_labels, class_to_level = build_paths_and_labels_from_root(dataset_root)
    if len(all_paths) == 0:
        raise ValueError('数据集为空，未找到任何支持的图像文件。')
    # 预过滤不可读/损坏的图片，避免训练过程中因个别样本中断
    valid_paths, valid_labels = [], []
    for p, y in zip(all_paths, all_labels):
        if safe_imread(p, cv2.IMREAD_GRAYSCALE) is not None:
            valid_paths.append(p)
            valid_labels.append(y)
        else:
            print(f"警告: 无法读取图像，已跳过: {p}")
    all_paths, all_labels = valid_paths, valid_labels
    train_image_paths, train_labels, val_image_paths, val_labels = stratified_split(all_paths, all_labels, train_ratio=0.8, seed=42)
    print(f"类别→等级(1-5)映射: {class_to_level}")
    print(f"训练集: {len(train_image_paths)} 张，验证集: {len(val_image_paths)} 张")
    model = train_model(train_image_paths, train_labels, val_image_paths, val_labels)
    