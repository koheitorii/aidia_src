import openpyxl

def save_dict_to_excel(data, file_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    headers = list(data.keys())
    sheet.append(headers)

    for row_data in zip(*[data[header] for header in headers]):
        sheet.append(row_data)

    workbook.save(file_path)