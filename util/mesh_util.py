def get_4_points_from_2_vert(point1, point3):
    point2 = [point1[0], point1[1], point3[2]]
    point4 = [point3[0], point3[1], point1[2]]
    return [point1, point2, point3, point4]


def get_4_points_from_2_horiz(point1, point3):
    point2 = [point3[0], point1[1], point3[2]]
    point4 = [point1[0], point3[1], point3[2]]
    return [point1, point2, point3, point4]
