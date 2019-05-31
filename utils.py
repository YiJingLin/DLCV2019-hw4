import matplotlib.pyplot as plt


def showFrames(row=1, column=5, frames):
    fig=plt.figure(figsize=(16,6))
    for idx, frame in enumerate(frames):
    fig.add_subplot(row,column,(idx+1)) # (row, column, idx)
    plt.imshow(frame)
    plt.show()