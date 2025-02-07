import pytest
from JuliaSet import calculate_z_serial_purepython


def test_julia_set_sum():
    # Constants from original code
    x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
    c_real, c_imag = -0.62772, -0.42193
    desired_width = 1000
    max_iterations = 300

    # Calculate step sizes
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width

    # Generate x coordinates
    x = []
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    # Generate y coordinates
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step

    # Build zs and cs lists
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    # Compute the output
    output = calculate_z_serial_purepython(max_iterations, zs, cs)

    # Assert the sum is correct
    assert sum(output) == 33219980


@pytest.mark.parametrize(
    "desired_width, max_iterations, expected_sum",
    [
        (1000, 300, 33219980),
        # So on and so forth...
    ]
)
def test_julia_set_sum_parametrized(desired_width, max_iterations, expected_sum):
    x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
    c_real, c_imag = -0.62772, -0.42193

    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width

    x = []
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step

    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    assert sum(output) == expected_sum