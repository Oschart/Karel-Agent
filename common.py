
class Action:
    move = 0
    turnLeft = 1
    turnRight = 2
    pickMarker = 3
    putMarker = 4
    finish = 5

    from_str = {
        "move": move,
        "turnLeft": turnLeft,
        "turnRight": turnRight,
        "pickMarker": pickMarker,
        "putMarker": putMarker,
        "finish": finish,
    }


class Direction:
    north = 0
    east = 1
    south = 2
    west = 3

    from_str = {
        "north": north,
        "east": east,
        "south": south,
        "west": west,
    }
