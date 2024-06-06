import random

import numpy as np
import open3d as o3d


def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points), pcd


def fit_line_ransac(points, max_iterations=10000, sigma=0.2, P=0.99):
    best_model = None
    best_inliers_count = 0
    best_inliers = []
    n_points = points.shape[0]

    for _ in range(max_iterations):
        # 随机选择两个点
        sample_indices = random.sample(range(n_points), 2)
        p1, p2 = points[sample_indices]

        # 在3d中定义直线方程
        direction = p2 - p1
        norm_dir = np.linalg.norm(direction)
        if norm_dir == 0:
            continue

        direction /= norm_dir
        cross_product = np.cross(direction, points - p1)

        # 计算所有点到直线的距离
        distances = np.linalg.norm(cross_product, axis=1)
        inliers = distances < sigma

        inliers_count = np.sum(inliers)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_model = (p1, direction)
            best_inliers = points[inliers]

        # 选择n个内点计算拟合出正确模型的概率
        w = inliers_count / n_points
        current_P = 1 - (1 - w ** 2) ** (_ + 1)
        if current_P > P:
            break

    return best_model, best_inliers_count, best_inliers


def draw_line(p1, direction, color=[1, 0, 0]):
    line_points = [p1 + direction * t for t in np.linspace(-100, 100, num=101)]
    line_pcd = o3d.geometry.PointCloud()
    line_pcd.points = o3d.utility.Vector3dVector(line_points)
    line_pcd.paint_uniform_color(color)
    return line_pcd, line_points


def line():
    file_path = 'RansacFitMutiLine3.pcd'
    points, pcd = read_pcd(file_path)

    #使用RANSAC拟合直线
    best_model, inliers_count, inliers_points = fit_line_ransac(points)

    if best_model is not None:
        p1, direction = best_model
        print(f"最好拟合线: 关键点 {p1}, 方向 {direction}")
        print(f"内点数目: {inliers_count}")

        #画出直线
        line_pcd, line_points = draw_line(p1, direction)

        # 点云、内点、拟合直线可视化
        inliers_pcd = o3d.geometry.PointCloud()
        inliers_pcd.points = o3d.utility.Vector3dVector(inliers_points)
        inliers_pcd.paint_uniform_color([0, 1, 0])  # Green for inliers

        # o3d.visualization.draw_geometries([pcd, inliers_pcd, line_pcd])
        # o3d.visualization.draw_geometries([inliers_pcd, line_pcd])
        # o3d.visualization.draw_geometries([inliers_pcd])
        o3d.visualization.draw_geometries([line_pcd])

        # # 输出内点的3D坐标
        # inliers_list = inliers_points.tolist()
        #np.savetxt('Inliners_Date.txt', inliers_list)
        # print("内点 3D 坐标:")
        # for point in inliers_list:
        #     print(point)

        # 输出直线上的3D坐标
        line_points_list = [point.tolist() for point in line_points]
        np.savetxt('Line_Date.txt', line_points_list)
        print("拟合直线 3D 坐标:")
        for point in line_points_list:
            print(point)
    else:
        print("没有拟合成直线.")


if __name__ == "__main__":
    line()

# RansacFitMutiLine4.pcd
# RansacFitMutiLine3.pcd
# Region_growing_cluster2.pcd
# Region_growing_cluster6.pcd
