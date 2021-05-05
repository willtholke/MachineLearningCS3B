class Side:
    UPSTREAM = 0
    DOWNSTREAM = 1


neighbors = {
    Side.UPSTREAM: [],
    Side.DOWNSTREAM: []
}

nodes = [1, 2, 3, 4]
neighbors[Side.UPSTREAM] = nodes

print(neighbors)

