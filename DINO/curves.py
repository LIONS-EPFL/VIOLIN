import numpy as np

def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))

def compute_hilbert_order(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return [(x, y) for x,y in gilbert2d(rows, cols)]

def interleave_bits(x, y):
    """
    Interleave the bits of two integers (x, y) to compute Morton order.
    """
    def split_bits(value):
        result = 0
        for i in range(32):  # Support up to 32-bit integers
            result |= ((value >> i) & 1) << (2 * i)
        return result

    return split_bits(x) | (split_bits(y) << 1)

def compute_morton_order(grid):
    """
    Compute Morton order for a grid.

    Parameters:
    - grid: A 2D list representing the grid.

    Returns:
    - A list of tuples containing Morton keys and coordinates (x, y).
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    morton_order = []

    for y in range(rows):
        for x in range(cols):
            morton_key = interleave_bits(x, y)
            morton_order.append((morton_key, x, y))

    # Sort by Morton key to achieve the Morton curve order
    morton_order.sort(key=lambda pair: pair[0])
    return [(x, y) for _, x, y in morton_order]

def coords_to_index(grid,coords):
    index = []
    for (x, y) in coords:
        index.append(int(grid[y][x]))
    return index

def index_to_coords_indexes(coords, grid_width, grid_height):
    indexes = [[None for _ in range(grid_width)] for _ in range(grid_height)]
    for i, (x, y) in enumerate(coords):
        indexes[y][x] = i
    return np.array(indexes).flatten()

def s_curve(grid):
    rows = len(grid)
    cols = len(grid[0])
    order = []
    for y in range(rows):
        if y % 2 == 0:
            # Left-to-right for even rows
            order.extend((x, y) for x in range(cols))
        else:
            # Right-to-left for odd rows
            order.extend((x, y) for x in reversed(range(cols)))
    return order
    
def compute_zigzag_order(grid):
    """Returns the elements of the grid in diagonal zig-zag order."""
    n_rows, n_cols = grid.shape
    indices = []
    for d in range(n_rows + n_cols - 1):
        if d % 2 == 0:
            r = min(d, n_rows - 1)
            c = d - r
            while r >= 0 and c < n_cols:
                indices.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(d, n_cols - 1)
            r = d - c
            while c >= 0 and r < n_rows:
                indices.append((r, c))
                c -= 1
                r += 1
    return indices

def compute_curve_order(grid, orientation):
    if orientation == 'snake':
        order = s_curve(grid)
    elif orientation == 'snake_T':
        order = s_curve(grid)   
        order = [(y,x) for x,y in order]       
    elif orientation == 'hilbert':  
        order = compute_hilbert_order(grid)        
    elif orientation == 'hilbert_T':  
        order = compute_hilbert_order(grid)   
        order = [(y,x) for x,y in order]
    elif orientation == 'peano':
        order = compute_morton_order(grid)
    elif orientation == 'peano_T':
        order = compute_morton_order(grid)
        order = [(y,x) for x,y in order]
    elif orientation == 'zigzag':
        order = compute_zigzag_order(grid)
    elif orientation == 'zigzag_T':
        order = compute_zigzag_order(grid)
        order = [(y,x) for x,y in order]
    return order
