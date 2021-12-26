from enum import Enum

class Action:
    move = 0
    turnLeft = 1
    turnRight = 2
    pickMarker = 3
    putMarker = 4
    finish = 5

    from_str = {
        'move': 0,
        'turnLeft' : 1,
        'turnRight' : 2,
        'pickMarker' : 3,
        'putMarker' : 4,
        'finish' : 5
    }


class Direction:
    north = 1
    east = 2
    south = 3
    west = 4