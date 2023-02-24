### Color mapping

In source repo was [table](https://github.com/ApolloScapeAuto/dataset-api/blob/master/lane_segmentation/helpers/laneMarkDetection.py) and [document](https://github.com/ApolloScapeAuto/dataset-api/blob/0afec45a123d04a6e0dbead0345c307ed7dc4482/lane_segmentation/LanemarkDiscription.pdf)

But RGB colors of the loaded masks are different (https://apolloscape.auto/lane_segmentation.html#to_dataset_href)

So below is its mapping:
    "SYD": (60, 15, 67),  # solid yellow dividing
    "BWG": (142, 35, 8),  # broken white guiding
    "SWD": (180, 173, 43),  # solid white dividing
    "SWS": (0, 0, 192),  # solid white stopping
    "CWYZ": (153, 102, 153)  # zebra
    # "AWR": (35, 136, 226),      # arrow white right turn
    # "ALW": (180, 109, 91),      # arrow white left turn
    # "AWTL": (160, 168, 234),    # arrow white thru & left turn
