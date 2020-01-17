import matplotlib.pyplot as plt



def set_position(fig=None, x=None, y=None):
    if fig is None:
        fig = plt.gcf()
    figManager = fig.canvas.manager.window
    w_screen, h_screen = get_screensize(fig)
    w, h = get_size(fig)
    if x is None:
        x = w_screen *1./4
    if y is None:
        y = (h_screen - h)//2
    # print(w_screen, w)
    figManager.geometry('+%d+%d' % ( x, y))


def set_size(fig=None, w=None,h=None, w_mul=1.0, h_mul=1.0):
    if fig is None:
        fig = plt.gcf()
    figManager = fig.canvas.manager.window
    if w is None:
        w = figManager.winfo_screenwidth()
    if h is None:
        h = figManager.winfo_screenheight()
    #print( '\n{}\n{}\n'.format(w, h))
    w*=w_mul
    h*=h_mul
    figManager.geometry("%dx%d" % (w, h))


def get_size( fig=None ):
    if fig is None:
        fig = plt.gcf()
    figManager = fig.canvas.manager.window
    return figManager.winfo_width(), figManager.winfo_height()

def get_screensize(fig=None):
    figManager = fig.canvas.manager.window
    return figManager.winfo_screenwidth(), figManager.winfo_screenheight()

def set_equal(fig=None):
    if fig is None:
        fig = plt.gca()
    fig.set_aspect('equal', adjustable='box')

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure()
    set_size(fig,100,100)
    # print( get_screensize(fig) )
    set_postion(fig)

    plt.show()
    exit()