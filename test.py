# from import_profiler import profile_import
# import time
# s = time.time()
#
# with profile_import() as context:
# # Anything expensive in here
#     import theano.tensor as T
#     from theano import function
# e = time.time()
# print(e - s)
#
# # Print cumulative and inline times. The number of + in the 3rd column
# # indicates the depth of the stack.
# context.print_info()

from import_profiler import profile_import

with profile_import() as context:
# Anything expensive in here
    from theano import *

# Print cumulative and inline times. The number of + in the 3rd column
# indicates the depth of the stack.
context.print_info()
