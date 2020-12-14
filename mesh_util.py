import trimesh


def get_4_points_from_2_vert(point1, point3):
    point2 = [point1[0], point1[1], point3[2]]
    point4 = [point3[0], point3[1], point1[2]]
    return [point1, point2, point3, point4]


def get_4_points_from_2_horiz(point1, point3):
    point2 = [point3[0], point1[1], point3[2]]
    point4 = [point1[0], point3[1], point3[2]]
    return [point1, point2, point3, point4]


# gets a list of 4 vertices of a rectangle and returns a mesh of it
def get_rectangle_mesh(point_list):
    for point in point_list:
        if point[0]>200 or point[0]<0:
            print("here")
        if point[1]>200 or point[1]<0:
            print("here")
    return trimesh.Trimesh(vertices=point_list, faces=[[0, 1, 2], [0, 2, 3]])
