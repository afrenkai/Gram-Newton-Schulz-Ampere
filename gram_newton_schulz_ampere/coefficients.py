def polar_express_coefficients() -> tuple[tuple[float, float, float], ...]:
    unmodified_coefficients = (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    )
    safety_factor = 1.05
    return tuple(
        (
            coefficient_one / safety_factor,
            coefficient_two / safety_factor**3,
            coefficient_three / safety_factor**5,
        )
        for coefficient_one, coefficient_two, coefficient_three in unmodified_coefficients
    )
