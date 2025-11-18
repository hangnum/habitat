import numpy as np
import SimpleITK as sitk
import nibabel as nib
from sklearn.cluster import KMeans
from pyradiomics.radiomics import featureextractor
from skimage import measure
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os

def load_medical_image(image_path, format_type='auto'):
    """
    通用的医学影像加载函数，支持多种格式
    
    Parameters:
    -----------
    image_path : str
        影像文件路径
    format_type : str
        格式类型，可选: 'auto', 'sitk', 'nibabel', 'dicom', 'nifti', 'nrrd'
    
    Returns:
    --------
    image_data : numpy.ndarray
        影像数据数组
    image_obj : object
        原始影像对象（SimpleITK或nibabel）
    """
    
    if format_type == 'auto':
        # 自动检测格式
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.nii', '.nii.gz']:
            format_type = 'nifti'
        elif file_ext == '.nrrd':
            format_type = 'nrrd'
        elif file_ext == '.dcm':
            format_type = 'dicom'
        elif file_ext in ['.mha', '.mhd']:
            format_type = 'sitk'
        else:
            # 默认使用SimpleITK
            format_type = 'sitk'
    
    print(f"使用 {format_type} 格式加载影像: {image_path}")
    
    if format_type == 'sitk' or format_type in ['nifti', 'nrrd', 'dicom']:
        # 使用SimpleITK加载（推荐用于PyRadiomics）
        image_obj = sitk.ReadImage(image_path)
        image_data = sitk.GetArrayFromImage(image_obj)
        return image_data, image_obj
    elif format_type == 'nibabel':
        # 使用nibabel加载（保持向后兼容）
        image_obj = nib.load(image_path)
        image_data = image_obj.get_fdata()
        return image_data, image_obj
    else:
        raise ValueError(f"不支持的格式类型: {format_type}")

def convert_image_format(input_path, output_path, input_format='auto', output_format='sitk'):
    """
    转换医学影像格式
    
    Parameters:
    -----------
    input_path : str
        输入文件路径
    output_path : str
        输出文件路径
    input_format : str
        输入格式
    output_format : str
        输出格式
    """
    print(f"转换影像格式: {input_path} -> {output_path}")
    
    # 加载影像
    image_data, image_obj = load_medical_image(input_path, input_format)
    
    if output_format == 'sitk':
        # 使用SimpleITK保存
        sitk.WriteImage(image_obj, output_path)
    elif output_format == 'nibabel':
        # 转换为nibabel格式并保存
        if isinstance(image_obj, sitk.SimpleITK.Image):
            # 从SimpleITK转换为nibabel
            affine = np.eye(4)  # 简单的单位矩阵，实际应用中需要从SimpleITK获取正确的仿射矩阵
            nib_image = nib.Nifti1Image(image_data, affine)
            nib.save(nib_image, output_path)
        else:
            nib.save(image_obj, output_path)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")
    
    print(f"格式转换完成: {output_path}")

# 配置影像格式
IMAGE_FORMAT = 'sitk'  # 可选: 'sitk', 'nibabel', 'auto'
IMAGE_PATH = './pyradiomics/data/brain1_image.nrrd'

# 可选：格式转换示例（如果需要转换格式，取消注释以下行）
# convert_image_format('./pyradiomics/data/brain1_image.nrrd', './brain1_image.nii.gz', 'sitk', 'nibabel')
# IMAGE_PATH = './brain1_image.nii.gz'

# 载入医学影像
image_data, image_obj = load_medical_image(IMAGE_PATH, IMAGE_FORMAT)

# 扁平化影像数据，以便KMeans聚类
image_flattened = image_data.flatten().reshape(-1, 1)

# 使用KMeans进行聚类
n_clusters = 3  # 假设分为3个类
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(image_flattened)

# 将聚类标签重新形状为图像的形状
labels_reshaped = labels.reshape(image_data.shape)

# 显示聚类后的标签图像
try:
    plt.imshow(labels_reshaped[:, :, labels_reshaped.shape[2] // 2], cmap='viridis')
    plt.title('KMeans Clustering Result')
    plt.colorbar()
    plt.show()
except Exception as e:
    print(f"无法显示图像（可能是无GUI环境）: {e}")
    # 保存图像到文件
    plt.imshow(labels_reshaped[:, :, labels_reshaped.shape[2] // 2], cmap='viridis')
    plt.title('KMeans Clustering Result')
    plt.colorbar()
    plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
    print("聚类结果已保存到 clustering_result.png")
    plt.close()

def create_sitk_mask(mask_array, reference_image):
    """
    将numpy掩码数组转换为SimpleITK图像对象
    
    Parameters:
    -----------
    mask_array : numpy.ndarray
        掩码数组
    reference_image : SimpleITK.Image
        参考影像，用于设置空间信息
    
    Returns:
    --------
    sitk_mask : SimpleITK.Image
        SimpleITK格式的掩码
    """
    # 确保掩码是整数类型
    mask_array = mask_array.astype(np.uint8)
    
    # 创建SimpleITK图像
    sitk_mask = sitk.GetImageFromArray(mask_array)
    
    # 复制参考影像的空间信息
    sitk_mask.CopyInformation(reference_image)
    
    return sitk_mask

# 初始化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor()

# 创建一个空的字典来存储每个聚类的特征
cluster_features = {}

# 对每个聚类区域提取特征
for cluster_id in range(n_clusters):
    # 创建一个二进制掩码（mask）只包含当前聚类
    cluster_mask_array = (labels_reshaped == cluster_id)
    
    # 确保聚类区域不是空的
    if np.sum(cluster_mask_array) == 0:
        continue
    
    # 将numpy掩码转换为SimpleITK格式
    cluster_mask_sitk = create_sitk_mask(cluster_mask_array, image_obj)
    
    # 从原始图像和聚类掩码中提取特征
    # 使用SimpleITK图像对象进行特征提取
    feature_result = extractor.execute(image_obj, cluster_mask_sitk)
    
    # 存储该聚类的特征
    cluster_features[cluster_id] = feature_result

# 打印每个聚类的特征
for cluster_id, features in cluster_features.items():
    print(f"Cluster {cluster_id} features:")
    for feature_name, feature_value in features.items():
        print(f"  {feature_name}: {feature_value}")
