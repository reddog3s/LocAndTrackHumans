import pandas as pd

keypoints = [
        'Nose',
        'Neck',
        'RShoulder',
        'RElbow',
        'RWrist',
        'LShoulder',
        'LElbow',
        'LWrist',
        'MidHip',
        'RHip',
        'RKnee',
        'RAnkle',
        'LHip',
        'LKnee',
        'LAnkle',
        'REye',
        'LEye',
        'REar',
        'LEar',
        'LBigToe',
        'LSmallToe',
        'LHeel',
        'RBigToe',
        'RSmallToe',
        'RHeel']


file_names = ['tossball0',
                'tossball1',
                'tossball2',
                'tossball3',
                'tossball4',
                'tossball5',
                'tossball6',
                'greetingchattingeating0',
                'greetingchattingeating1',
                'greetingchattingeating2',
                'greetingchattingeating3',
                'greetingchattingeating4',
                'greetingchattingeating5',
                'greetingchattingeating6',
                'greetingchattingeating7',
                'dips0',
                'dips4',
                'dips6',
                'reparingprojector0',
                'reparingprojector3',
                'reparingprojector4',
                'reparingprojector5',
                'burpeejump0',
                'burpeejump1',
                'burpeejump2',
                'burpeejump3']




results = pd.DataFrame()
for file in file_names:
    temp = pd.read_csv('eval/' + file + '.csv')
    results = pd.concat([results, temp])

print(results['nazwa'].unique())

print(results)
print(results.mean())

res_for_keypoints = pd.DataFrame(columns=keypoints)
print('\n')
for name in keypoints:
    temp = results[results['punkt'] == name][['PCK', 'AP', 'AvgErr']]
    res_for_keypoints[name] = temp.mean()
res_for_keypoints = res_for_keypoints.T

print('res for keypoints')
print(res_for_keypoints.mean())
res_for_keypoints.to_csv('res_for_keypoints.csv')


head = results[results['punkt'] == 'Nose']
head = pd.concat([head, results[results['punkt'] == 'REye']])
head = pd.concat([head, results[results['punkt'] == 'LEye']])
head = pd.concat([head, results[results['punkt'] == 'REar']])
head = pd.concat([head, results[results['punkt'] == 'LEar']])


seqs = results['nazwa'].unique()
pcks = pd.DataFrame(columns=['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'mean'], index = seqs)
aps = pd.DataFrame(columns=['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'mean'], index = seqs)
for seq_name in seqs:
    seq = results[results['nazwa'] == seq_name]

    #oblicz glowa
    head = seq[seq['punkt'] == 'Nose']
    head = pd.concat([head, seq[seq['punkt'] == 'REye']])
    head = pd.concat([head, seq[seq['punkt'] == 'LEye']])
    head = pd.concat([head, seq[seq['punkt'] == 'REar']])
    head = pd.concat([head, seq[seq['punkt'] == 'LEar']])
    pcks.loc[seq_name, 'Head'] = head['PCK'].mean()
    aps.loc[seq_name, 'Head'] = head['AP'].mean()

    shoulder = seq[seq['punkt'] == 'RShoulder']
    shoulder = pd.concat([shoulder, seq[seq['punkt'] == 'LShoulder']])
    pcks.loc[seq_name, 'Shoulder'] = shoulder['PCK'].mean()
    aps.loc[seq_name, 'Shoulder'] = shoulder['AP'].mean()
    

    elbow = seq[seq['punkt'] == 'RElbow']
    elbow = pd.concat([elbow, seq[seq['punkt'] == 'LElbow']])
    pcks.loc[seq_name, 'Elbow'] = elbow['PCK'].mean()
    aps.loc[seq_name, 'Elbow'] = elbow['AP'].mean()


    wrist = seq[seq['punkt'] == 'RWrist']
    wrist = pd.concat([wrist, seq[seq['punkt'] == 'LWrist']])
    pcks.loc[seq_name, 'Wrist'] = wrist['PCK'].mean()
    aps.loc[seq_name, 'Wrist'] = wrist['AP'].mean()

    knee = seq[seq['punkt'] == 'RKnee']
    knee = pd.concat([knee, seq[seq['punkt'] == 'LKnee']])
    pcks.loc[seq_name, 'Knee'] = knee['PCK'].mean()
    aps.loc[seq_name, 'Knee'] = knee['AP'].mean()

    ankle = seq[seq['punkt'] == 'RAnkle']
    ankle = pd.concat([ankle, seq[seq['punkt'] == 'LAnkle']])
    pcks.loc[seq_name, 'Ankle'] = ankle['PCK'].mean()
    aps.loc[seq_name, 'Ankle'] = ankle['AP'].mean()

    hip = seq[seq['punkt'] == 'RHip']
    hip = pd.concat([hip, seq[seq['punkt'] == 'LHip']])  
    hip = pd.concat([hip, seq[seq['punkt'] == 'MidHip']]) 
    pcks.loc[seq_name, 'Hip'] = hip['PCK'].mean()
    aps.loc[seq_name, 'Hip'] = hip['AP'].mean()

    pcks.loc[seq_name, 'mean'] = pcks.loc[seq_name].mean()
    aps.loc[seq_name, 'mean'] = aps.loc[seq_name].mean()   


print(pcks.mean())  

print(aps.mean())
pcks.to_csv('pcks.csv')
aps.to_csv('aps.csv')