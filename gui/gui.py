import PySimpleGUI as sg
import os
from utils import get_mnist_dataset
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use("TkAgg")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


import matplotlib.pyplot as plt
from attack import  attack_single_image, gui_call_evaluate_attack


def get_borderless_figure(size=(4, 4)):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax


"""
Pick folder:
[
    sg.Text("Image Folder"),
    sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
    sg.FolderBrowse(initial_folder='trained_models'),
],
"""

folder = 'trained_models'
try:
    # Get list of files in folder
    file_list = os.listdir(folder)
except:
    file_list = []

models = [
    f[:-4]
    for f in file_list
    if os.path.isfile(os.path.join(folder, f))
       and f.lower().endswith((".pth"))
]
assert len(models) > 0, "no files"
file_list_column = [
    [sg.Text("Pretrained Models:")],
    [sg.Listbox(values=models, enable_events=True, size=(40, 20), key="-MODELS-")],

    [sg.Text("Available Attacks:")],
    [sg.Listbox(values=['Additive Noise', '0 - 1', 'Complementary'], enable_events=True, size=(40, 20),
                key="-ATTACKS-")],
]

image_column_1 = [
    [sg.Text("Original")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE1-")],
]

image_column_2 = [
    [sg.Text("Attacked")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE2-")],
]

slider = [
    sg.Text('Number of Pixels'),
    sg.Slider(range=(1, 28 * 28),
              enable_events=True,
              default_value=10,
              size=(50, 15),
              orientation='horizontal', key="-SLIDER-")
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_column_1),
        sg.Column(image_column_2),
    ],
    [
        sg.Column([
            slider,
            [sg.Button("Execute Attack", key='-ATTACKBUTTON-')],
        ]),
        sg.Column([
           [sg.Text('', key='-INFOTEXT-')],
        ])
     ]


]

# layout = [[sg.Text("test text")], [sg.Button("Ok")]]

window = sg.Window(title="Pixel Attacks", layout=layout)

dataset = None
model = None
trained_model_name = None
selected_attack = None

fig, ax = get_borderless_figure()
image = None
k = 10
mc = False


def attack_change():
    if selected_attack == 'Additive Noise':
        attacked = attack_single_image(image, 'additive_noise', k, 123)
    elif selected_attack == '0 - 1':
        attacked = attack_single_image(image, 'zero_one', k, 123)
    elif selected_attack == 'Complementary':
        attacked = attack_single_image(image, 'complementary', k, 123)

    ax.imshow(attacked, vmin=0, vmax=1)
    fig.savefig('gui/images/attacked.png')
    window["-IMAGE2-"].update(filename='gui/images/attacked.png')


def image_change():
    ax.imshow(image, vmin=0, vmax=1)
    fig.savefig('gui/images/original.png')
    window["-IMAGE1-"].update(filename='gui/images/original.png')
    if selected_attack:
        attack_change()

    print('plot attack')


while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button

    if event == '-MODELS-':
        trained_model_name = values[event][0]

        s = trained_model_name.split('_')

        model, dataset = s[0], s[1]
        if len(s) > 2:
            if s[2].endswith('mcdropout'):
                mc = True
            else:
                mc = False
        else: mc = False


        dataset = get_mnist_dataset(dataset, 28)
        image = dataset.imgs[0] / 255.0
        image_change()

    elif event == '-ATTACKS-':
        selected_attack = values[event][0]
        if dataset:
            if image is not None:
                attack_change()
            else:
                image_change()

    elif event == '-SLIDER-':
        k = int(values[event])
        if image is not None and selected_attack is not None:
            attack_change()
    elif event == '-ATTACKBUTTON-':
        model = values['-MODELS-']
        attack = values['-ATTACKS-']
        num_pixels = values['-SLIDER-']

        window['-INFOTEXT-'].update('Loading ...')
        window.refresh()
        print('-----------------------------')
        print(model, dataset, mc)
        print('-----------------------------')

        if selected_attack == 'Additive Noise':
            test_metrics, attack_metrics = gui_call_evaluate_attack(trained_model_name, dataset.flag, int(num_pixels), 'additive_noise', mc=mc)
        elif selected_attack == '0 - 1':
            test_metrics, attack_metrics = gui_call_evaluate_attack(trained_model_name, dataset.flag, int(num_pixels), 'zero_one', mc=mc)
        elif selected_attack == 'Complementary':
            test_metrics, attack_metrics = gui_call_evaluate_attack(trained_model_name, dataset.flag, int(num_pixels), 'complementary', mc=mc)
        text = f'Original: acc: {round(test_metrics[1],3)} auc: {round(test_metrics[0],3)}\n\n' \
               f'Attacked: acc: {round(attack_metrics[1],3)} auc: {round(attack_metrics[0],3)}'
        window['-INFOTEXT-'].update(text)

    elif event == "OK" or event == sg.WIN_CLOSED:
        break
    else:
        print(values)
