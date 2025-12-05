import csv
import json
import filecmp
import difflib
import os.path
import pathlib


def write_csv(filename_, content):
    with open(filename_, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(content)


def load_json(file__name):
    if os.path.isfile(file__name):
        data_file = open(file__name, "r", encoding='utf-8')
        file_data = json.loads(data_file.read())
        data_file.close()
        return file_data
    else:
        return {}


def write_json(file__name, content):
    with open(file__name, "w", encoding="utf-8") as text_file:
        print(json.dumps(content, indent=4), file=text_file)


def write_text(file__name, content):
    with open(file__name, "a", encoding="utf-8") as text_file:
        print(content, file=text_file)


def write_jsonl(file__name, content):
    with open(file__name, "a", encoding="utf-8") as text_file:
        print(json.dumps(content), file=text_file)


def compare_files(file1, file2):
    return filecmp.cmp(file1, file2)


def differences(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        diff = difflib.unified_diff(f1.readlines(),
                                    f2.readlines(), fromfile=file1, tofile=file2)
    str_diff = [str(x) for x in diff]
    # for line in diff:
    #     print(line)
    return str_diff


def get_dir_list(path):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
        print(path.name)
    file_list = []
    for item in path.iterdir():
        if item.is_file():
            file_list.append(item)
        if item.is_dir():
            file_list += get_dir_list(item)
    return file_list


def compare(dir1, dir2):
    result = {"not_equals": [], "differences": []}
    dir_list1 = get_dir_list(dir1)
    str_list1 = [str(x).replace("\\", "/").replace(dir1+"/", "") for x in dir_list1]
    dir_list2 = get_dir_list(dir2)
    str_list2 = [str(x).replace("\\", "/").replace(dir2+"/", "") for x in dir_list2]
    not_common_1 = []
    for i, item in enumerate(str_list1):
        if item not in str_list2:
            print(item)
            not_common_1.append(item)
        else:
            file1 = dir1+"/"+item
            file2 = dir2+"/"+item
            comp = compare_files(file1, file2)
            if not comp:
                result["not_equals"].append(item)
                if file1.endswith("js"):
                    write_text("differences.txt", (differences(file1, file2)))
                    result["differences"].append({"filename": item, "diff": str(differences(file1, file2))})

    print()
    not_common_2 = []
    for i, item in enumerate(str_list2):
        if item not in str_list1:
            print(item)
            not_common_2.append(item)
    result["not_common_1"] = not_common_1
    result["not_common_2"] = not_common_2
    write_json("dir_comparing_results.json", result)



