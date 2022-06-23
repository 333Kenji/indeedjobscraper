import numpy as numpy
import matplotlib.pyplot as plt

def to_img(data, name, col_width=4.0, row_height=0.625, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    
    if ax is None:
        size = (numpy.array(data.shape[::-1]) + numpy.array([0, 1])) * numpy.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(True)
    mpl_table.set_fontsize(font_size)
    """
    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    """
    fig = ax.get_figure()
    fig.savefig(f'../Images/{name}.jpg')
    plt.close()
    return