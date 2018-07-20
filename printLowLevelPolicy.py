directory = r'C:/Projects/277/platform-ll-main-mcu/functional-code/application/trac-motors/'

# This just for testing
from policy import TilePol
import numpy as np

def exportPolicy(pol, w):
    header = open(directory + 'policy_table.h', 'w')
    header.write("#ifndef POLICY_TABLE_H\n#define POLICY_TABLE_H\n\n")
    header.write("#include \"types.h\"\n\n")
    header.write("int8_t table_get_dX(void);\n")
    header.write("int8_t table_get_nX();\n")
    header.write("int8_t table_get_startX();\n")
    header.write("int16_t table_get_minU();\n")
    header.write("int16_t table_get_maxU();\n")
    header.write("int16_t table_getIndex(int8_t indx);\n")
    header.write("\n#endif\n")
    header.close()

    dotc = open(directory + 'policy_table.c', 'w')
    dotc.write("#include \"policy_table.h\"\n\n")
    dotc.write('static int16_t table[{}] = {{'.format(int(pol.nX)))
    for i in xrange(pol.nX-1):
        dotc.write('{}, '.format(int(w[i])))
    dotc.write('{}}};\n\n'.format(int(w[pol.nX-1])))

    dotc.write("int8_t table_get_dX(void)\n{\n")
    dotc.write('\treturn {};\n}}\n\n'.format(int(pol.deltaX)))
    dotc.write("int8_t table_get_nX()\n{\n")
    dotc.write('\treturn {};\n}}\n\n'.format(int(pol.nX)))
    dotc.write("int8_t table_get_startX()\n{\n")
    dotc.write('\treturn {};\n}}\n\n'.format(int(pol.startX)))
    dotc.write("int16_t table_get_minU()\n{\n")
    dotc.write('\treturn {};\n}}\n\n'.format(int(pol.min)))
    dotc.write("int16_t table_get_maxU()\n{\n")
    dotc.write('\treturn {};\n}}\n\n'.format(int(pol.max)))
    dotc.write("int16_t table_getIndex(int8_t indx)\n{\n\treturn table[indx];\n}\n\n")

    dotc.close()

if __name__ == "__main__":
    pol = TilePol(21, 10, -32766, 32766, -100)
    ws = 2000 * np.ones(21)
    exportPolicy(pol, ws)
