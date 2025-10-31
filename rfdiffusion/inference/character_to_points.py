# rfdiffusion/inference/character_to_points.py
import numpy as np

def char_to_3d_points(character, scale=10.0, density=5):
    """
    将字符转换为3D点云表示
    
    Args:
        character: 输入字符 (A-Z, 0-9)
        scale: 字符大小缩放因子
        density: 点密度
    
    Returns:
        numpy array of shape (N, 3) representing 3D points
    """
    points = []
    
    # 示例：简单几何形状代表不同字符
    if character.upper() == 'O':
        # 生成圆形点云
        for i in range(density * 10):
            angle = 2 * np.pi * i / (density * 10)
            x = scale * np.cos(angle)
            y = scale * np.sin(angle)
            z = 0
            points.append([x, y, z])
            
    elif character.upper() == 'I':
        # 生成直线点云
        for i in range(density * 5):
            x = 0
            y = scale * (i / (density * 5) - 0.5)
            z = 0
            points.append([x, y, z])
            
    elif character.upper() == 'L':
        # 生成L形点云
        # 垂直线
        for i in range(density * 5):
            x = -scale/2
            y = scale * (i / (density * 5) - 0.5)
            z = 0
            points.append([x, y, z])
        # 水平线
        for i in range(density * 3):
            x = -scale/2 + scale * i / (density * 3)
            y = -scale/2
            z = 0
            points.append([x, y, z])
            
    elif character.upper() == 'H':
        # H 形状
        # 左垂直线
        for i in range(density * 5):
            x = -scale/3
            y = scale * (i / (density * 5) - 0.5)
            z = 0
            points.append([x, y, z])
        # 右垂直线
        for i in range(density * 5):
            x = scale/3
            y = scale * (i / (density * 5) - 0.5)
            z = 0
            points.append([x, y, z])
        # 中间横线
        for i in range(density * 3):
            x = -scale/3 + (2*scale/3) * i / (density * 3)
            y = 0
            z = 0
            points.append([x, y, z])
            
    elif character.upper() == 'A':
        # A 形状（三角形加横线）
        # 两边
        for i in range(density * 5):
            ratio = i / (density * 5)
            x = -scale/2 + ratio * scale
            y = -scale/2 + abs(x) * 2
            z = 0
            points.append([x, y, z])
        # 中间横线
        for i in range(density * 3):
            x = -scale/4 + (scale/2) * i / (density * 3)
            y = 0
            z = 0
            points.append([x, y, z])
    else:
        # 默认圆圈形状
        for i in range(density * 10):
            angle = 2 * np.pi * i / (density * 10)
            x = scale * np.cos(angle)
            y = scale * np.sin(angle)
            z = 0
            points.append([x, y, z])
            
    return np.array(points)

def text_to_3d_points(text, spacing=15.0):
    """
    将文本转换为3D点云
    
    Args:
        text: 输入文本
        spacing: 字符间距
    
    Returns:
        numpy array of shape (N, 3) representing 3D points
    """
    all_points = []
    for i, char in enumerate(text):
        char_points = char_to_3d_points(char)
        # 添加字符间偏移
        char_points[:, 0] += i * spacing
        all_points.append(char_points)
    
    return np.vstack(all_points)