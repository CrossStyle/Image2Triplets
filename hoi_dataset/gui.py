from xml.dom.minidom import parse
import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import json

VOC_ROOT = 'VOCdevkit/VOC2007/Annotations'


def show_classname(save_path):
    name_set = set()
    with os.scandir(save_path) as entries:
        for entry in entries:
            # tmp.append(entry.name)
            temp_path = save_path + '/' + entry.name
            dom_tree = parse(temp_path)
            root_node = dom_tree.documentElement
            objects = root_node.getElementsByTagName("object")
            for obj in objects:
                name = obj.getElementsByTagName("name")[0]
                name_set.add(name.childNodes[0].data)
    print(name_set)
    return name_set


def show_all_boxes():
    global image_file
    global img
    global boxes_dict
    global img_path
    # img_path = filedialog.askopenfilename()
    text.insert(END, '\n', 'bold_italics')
    text.insert(END, 'Open photo 打开图片：' + str(img_path)+'\n', 'bold_italics')
    text.see("end")
    xml_path = VOC_ROOT + '/' + img_path.split('/')[-1][:-4] + '.xml'
    dom_tree = parse(xml_path)
    root_node = dom_tree.documentElement
    objects = root_node.getElementsByTagName("object")
    i = 0
    boxes_dict = {}
    img = cv2.imread(img_path)
    for obj in objects:
        name = obj.getElementsByTagName("name")[0].childNodes[0].data
        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)) + 3
        ymin = int(np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)) + 3
        xmax = int(np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)) - 1
        ymax = int(np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)) - 1
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, name + ' ' + str(i), (xmin + 5, ymin + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        boxes_dict[i] = [name, xmin, ymin, xmax, ymax]
        i += 1
    image_file = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(img, (800, 800)), cv2.COLOR_BGR2RGB)))
    canvas.create_image(400, 0, anchor='n', image=image_file)
    lb1.delete(0, END)
    for item in range(i):
        lb1.insert(END, str(boxes_dict[item][0]) + str(item))


def show_selected_boxes():
    global image_file
    global img
    global boxes_dict
    global box_index1
    box_index1 = lb1.curselection()
    # print(box_index1)
    # lb1.curselection() lb1.get()
    if len(box_index1) >= 2:
        for i in box_index1:
            xmin, ymin, xmax, ymax = boxes_dict[i][1:]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    else:
        xmin, ymin, xmax, ymax = boxes_dict[box_index1[0]][1:]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    image_file = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(img, (800, 800)), cv2.COLOR_BGR2RGB)))
    canvas.create_image(400, 0, anchor='n', image=image_file)


def clear_boxes():
    global img
    global boxes_dict
    global image_file
    for i in boxes_dict.keys():
        xmin, ymin, xmax, ymax = boxes_dict[i][1:]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    image_file = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(cv2.resize(img, (800, 800)), cv2.COLOR_BGR2RGB)))
    canvas.create_image(400, 0, anchor='n', image=image_file)


def print_info():
    box_info1, box_info2 = ' ', ' '
    tmp = []
    for i in lb1.curselection():
        box_info1 += '  ' + lb1.get(i)
        tmp.append(i)
    for i in lb2.curselection():
        box_info2 += '  ' + lb2.get(i)
        tmp.append(str(lb2.get(i)))
    # box_info2 = lb2.get(lb2.curselection())
    # tmp.append(str(box_info2))
    verb.append(tmp)
    text.insert(END, '\n', 'bold_italics')
    text.insert(END,  'Preview 预览： ', 'bold_italics')
    text.insert(END, str(box_info1)+' -->  ', 'bold_italics')
    text.insert(END, str(box_info2) + ' \n', 'bold_italics')
    text.see("end")


def select_output_path():
    global out_path
    out_path = 'finial.json'
    # out_path = filedialog.askopenfilename()
    text.insert(END, '\n', 'bold_italics')
    text.insert(END, 'Save path 保存路径：' + str(out_path)+'\n', 'bold_italics')
    text.see("end")


def export_annotation():
    with open(out_path, 'a') as f:
        f.write(img_path.split('/')[-1][:-4] + '.jpg ')
        for i in lb1.curselection():
            for j in boxes_dict[i]:
                f.write(str(j))
                f.write(' ')
            f.write(' ')
        f.write(str(lb2.get(lb2.curselection())))
        f.write('\n')
    text.insert(END, '\n', 'bold_italics')
    text.insert(END, 'Save successfully 保存成功！\n', 'bold_italics')
    text.see("end")


def export_json():
    global boxes_dict
    global img_path
    global verb
    img_name = img_path.split('/')[-1][:-4] + '.jpg'
    out_dict = {'img': img_name, 'boxes': boxes_dict, 'verb': verb}
    verb = []
    with open(out_path, 'a') as f:
        json.dump(out_dict, f, ensure_ascii=False)
        f.write('\n')
    text.insert(END, '\n', 'bold_italics')
    text.insert(END, 'Save successfully 保存成功！\n', 'bold_italics')
    text.see("end")


def show_next_img():
    global img_path
    img_file_path = os.path.dirname(img_path)
    all_img_file = os.listdir(img_file_path)
    img_index = all_img_file.index(img_path.split('/')[-1])
    if img_index + 1 <= len(all_img_file):
        img_path = img_file_path + '/' + all_img_file[img_index + 1]
        show_all_boxes()
    else:
        text.insert(END, '\n', 'bold_italics')
        text.insert(END, 'It is already the last picture! 已经是最后一张图片，到底啦！！！\n', 'bold_italics')
        text.see("end")


def show_previous_img():
    global img_path
    img_file_path = os.path.dirname(img_path)
    all_img_file = os.listdir(img_file_path)
    img_index = all_img_file.index(img_path.split('/')[-1])
    if img_index - 1 >= 0:
        img_path = img_file_path + '/' + all_img_file[img_index - 1]
        show_all_boxes()
    else:
        text.insert(END, '\n', 'bold_italics')
        text.insert(END, 'It is already the first photo to the top! 已经是第一张图片，到顶啦！！！\n', 'bold_italics')
        text.see("end")


if __name__ == '__main__':
    img_path = 'VOCdevkit/VOC2007/JPEGImages/1%20(1).mp4#t=12.jpg'
    image_file, img, boxes_dict, out_path = None, None, None, None

    verb = []

    win = tk.Tk()
    win.title('HOI Annotation Application')
    win.geometry('1340x800')
    win.resizable(width=False, height=False)

    # 画布
    canvas = Canvas(win, bg='green', height=800, width=800)
    canvas.grid(column=1, rowspan=4)

    # 日志窗
    text = Text(win, height=15, width=60)
    text.insert(END, 'Log output：\n', 'bold_italics')
    text.grid(row=3, column=2, columnspan=2)

    # 按钮窗
    b1 = Button(win, text='Input path', font=('Arial', 12), width=10, height=1, command=show_all_boxes)
    b1.grid(row=0, column=0)
    b2 = Button(win, text='save path', font=('Arial', 12), width=10, height=1, command=select_output_path)
    b2.grid(row=1, column=0)
    b3 = Button(win, text='Last one', font=('Arial', 12), width=10, height=1, command=show_previous_img)
    b3.grid(row=2, column=0)
    b4 = Button(win, text='Next one', font=('Arial', 12), width=10, height=1, command=show_next_img)
    b4.grid(row=3, column=0)

    b5 = Button(win, text='Show', font=('Arial', 12), width=10, height=1, command=show_selected_boxes)
    b5.grid(row=1, column=2)
    b6 = Button(win, text='Save', font=('Arial', 12), width=10, height=1, command=export_json)
    b6.grid(row=2, column=3)
    b7 = Button(win, text='Print', font=('Arial', 12), width=10, height=1, command=print_info)
    b7.grid(row=2, column=2)
    b8 = Button(win, text='Clear', font=('Arial', 12), width=10, height=1, command=clear_boxes)
    b8.grid(row=1, column=3)

    # List box
    var1 = StringVar()
    var2 = StringVar()
    var2.set(('敲打', '拿着',  '使用', '清扫', '回填', '推拉', '涂抹',  '搬运', '铺装', '驾驶', '搅拌', '铲',  '冷弯', '踩', '填塞',
              '绑扎', '切割', '打磨', '焊接', '清理', '安装', '运送', '0', '00', '000'))
    lb1 = tk.Listbox(win, listvariable=var1, selectmode=MULTIPLE, height=25, exportselection=0)
    lb1.grid(row=0, column=2)
    lb2 = tk.Listbox(win, listvariable=var2, height=25, exportselection=0, selectmode=MULTIPLE)
    lb2.grid(row=0, column=3)
    win.mainloop()

