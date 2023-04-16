import typing
import src.utils as utils

def orientation(pivot, q, r) -> str:
    """Find orientation from q to r about a pivot point.
        returns: 'cw', 'ccw', or 'col' (colinear)
    """
    val = (q[1] - pivot[1]) * (r[0] - q[0]) - (q[0] - pivot[0]) * (r[1] - q[1])
    if val == 0:
        return 'col'
    else:
        return 'cw' if val > 0 else 'ccw'

def compute(points: typing.List[typing.Tuple[float, float]]) -> typing.List[typing.Tuple[float, float]]:
    if len(points) < 3:
        return list(points)

    res = []
    leftmost_idx = min((idx for idx in range(len(points))), key=lambda _idx: points[_idx][0])
    pivot_idx = leftmost_idx

    while True:
        res.append(pivot_idx)  # add current pivot to hull
        next_idx = (pivot_idx + 1) % len(points)

        # find next pivot
        for i in range(len(points)):
            ori = orientation(points[pivot_idx], points[i], points[next_idx])
            if ori == 'ccw':
                next_idx = i
            elif ori == 'col':
                if utils.dist2(points[pivot_idx], points[i]) > utils.dist2(points[pivot_idx], points[next_idx]):
                    # If they're colinear, take the point that's farther away.
                    # This will eliminate extraneous edge points from the hull.
                    next_idx = i

        pivot_idx = next_idx  # continue from new pivot

        # if we've wrapped back around to the original pivot, we've completed the hull.
        if (pivot_idx == leftmost_idx):
            break

    return [points[idx] for idx in res]
