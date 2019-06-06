import matplotlib.pyplot as plt

def showFrames(frames, row=1, column=5):
    fig=plt.figure(figsize=(16,6))
    for idx, frame in enumerate(frames):
        fig.add_subplot(row,column,(idx+1)) # (row, column, idx)
        plt.imshow(frame)
    plt.show()