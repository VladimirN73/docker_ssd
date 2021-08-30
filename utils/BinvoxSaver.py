import numpy as np

def write(data, filename):

    fp = open(filename, 'wb')

    fp.write(b'#binvox 1\n')
    fp.write(b'dim 64 64 64\n')
    fp.write(b'translate 0 0 0\n')
    fp.write(b'scale 10\n')
    fp.write(b'data\n')

    state = data[0][0][0]
    if state == 1:
        state = True
    else:
        state = False

    ctr = 0
    for x in range(64):  # depth from back to front
        for y in range(64):  # from bottom to top
            for z in range(64):
                val = data[x][y][z]
                if val == 1:
                    val = True
                else:
                    val = False

                if val == state:
                    ctr += 1
                    # if ctr hits max, dump
                    if ctr == 63:
                        fp.write(chr(state).encode('ascii'))
                        fp.write(chr(ctr).encode('ascii'))
                        ctr = 0
                else:
                    # if switch state, dump
                    fp.write(chr(state).encode('ascii'))
                    fp.write(chr(ctr).encode('ascii'))
                    state = val
                    ctr = 1
            # flush out remainders
    if ctr > 0:
        fp.write(chr(state).encode('ascii'))
        fp.write(chr(ctr).encode('ascii'))
