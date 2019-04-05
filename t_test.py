dictionary_list = {'log_list': ["CYLINDERS",'fuck'],
                   'sqrt_list': ["WEIGHT"],
                   'cubic_list': ["ACCELERATION"]}

for key,value in dictionary_list.items():
    if key =='log_list':
        for val in value:
            print(val)
        # print('log list done', value)
    elif key=='sqrt_list':
        print('sqrt', value)
    elif key=='cubic_list':
        print('cube')
    else:
        print('done')