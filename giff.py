import imageio

filenames = []
for t in range(45):
    filenames.append('C:\pythonGPREPS\w1-' + str(t) + '.png')

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('C:\pythonGPREPS\w1.gif', images)
