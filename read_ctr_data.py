import gzip
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from collections import Counter
from datetime import datetime

###########
# Load data
f = gzip.open('train.gz', 'rb')

############
# Parameters

# number of minimum visits per user
min_vis = 2

# Time step cutoff
cutoff = 3


##########
# Read data

l = 0
data = []
for line in f:
    line = line.decode("utf-8")
    clean = line.replace('\r\n', '')

    clean = clean.split(',')
    data.append(clean)

    l += 1
    if l > 2000000:
        break
print('Length of data:', l)

# Only retain relevant columns
df = pd.DataFrame(data=data[1:], columns=data[0])
df = df.drop(['banner_pos', 'id', 'site_domain', 'app_domain', 'device_type', 'device_conn_type',
              'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], axis=1)

# Check which device_ips, a proxy for visitors, occur more than once
vc = df.device_ip.value_counts()
to_keep = set()
for v, c in vc.items():
    if c > 1:
        to_keep.add(v)

print('Total clicks:', len(to_keep))
df = df[df['device_ip'].isin(to_keep)]

print('Unique sites: ', df.site_id.nunique())
print('Unique devices: ', df.device_model.nunique())


###################################
# Check most frequent pages visited

page_list = list(df.site_id.values)
page_counter = Counter(page_list)
most_common = {}
no_ = 0
for pc in page_counter.most_common(50):
    most_common[pc[0]] = no_
    no_ += 1
print('Most common:', no_)
print(most_common)

# Only retain samples where there are more than the minimum number of visits
# Hour designates the session number per visitor (different hour = different session)
print('Filtering ('+str(len(df))+')')
df = df.groupby('device_ip').filter(lambda x: x['hour'].nunique() >= min_vis)
print('Done filtering ('+str(len(df))+')')

print(df.head(100))

# Encoding
channel_enc = {}
channel_e = 1
site_enc = {}
site_e = 1
device_enc = {}
device_e = 1

strings = []
labels = []
hits = []
time_diffs = []
avg_length = 0

to_keep = df.device_ip.values

# Run through all visitors
start_time = datetime. now()
for i, ip in enumerate(to_keep):

    # Print statistics
    if i % 100 == 0:
        current_time = datetime.now()
        duration = current_time - start_time

        print('i:', i, '/', len(to_keep), 'in', duration.total_seconds(), 'sec.')

    if i > 100000:
        break

    df_ip = df[df.device_ip == ip]
    avg_length += df_ip.hour.nunique()

    sessions = []
    label = []
    hits_h = []
    diffs = []

    first_row = 0
    first = True

    # Loop through the sessions (different hours) per visitor
    visit_no = 1
    for h, value in enumerate(df_ip.hour.unique()):
        # Store time difference
        if first:
            first_row = int(value)
            first = False
            diffs.append(0)
        else:
            diffs.append(int(value) - first_row)

        if len(sessions) == cutoff:
            break

        session = []
        df_ip_hour = df_ip[df_ip.hour == value]

        empty_arr = np.zeros(len(most_common))
        c_label = 0

        # Store web page visit info for this session
        # Most frequent page visits (yes/no) are stored in the empty_arr
        for r, row in df_ip_hour.iterrows():
            # Store the session info on device and channel
            if session == []:
                app = row.app_id
                device = row.device_model
                if device not in device_enc.keys():
                    device_enc[device] = device_e
                    device_e += 1
                if app not in channel_enc.keys():
                    channel_enc[app] = channel_e
                    channel_e += 1
                session = [channel_enc[app], device_enc[device], visit_no, 0]
                if int(row.click) == 1:
                    c_label = 1

            site = row['site_id']
            if site in most_common.keys():
                empty_arr[most_common[site]] = 1

        visit_no += 1
        hits_h.append(empty_arr)
        # diffs.append(0)
        sessions.append(session)
        label.append(int(c_label))

    hits.append(hits_h)
    strings.append(sessions)
    labels.append(label)
    time_diffs.append(diffs)

# Avg. number of page visits per visitor
print('Avg. length:', (avg_length/len(strings)))
prefix = 'ctr_'
parameter_string = str(min_vis)

print('#Channels:', len(channel_enc))

# pad documents to a max length of 4 words
padded_channels = pad_sequences(strings, maxlen=cutoff, padding='pre')
padded_time = pad_sequences(time_diffs, maxlen=cutoff, padding='pre')
padded_labels = pad_sequences(labels, maxlen=cutoff, padding='pre')

# Pad the hits as well
padded_hits = []
hit_dim = int(len(hits[0][0]))
for h, hit in enumerate(hits):
    if len(hit) >= cutoff:
        padded_hits.append(np.array(hit[-cutoff:]))
    else:
        hit_pad = []
        for i in range(0, cutoff - len(hit)):
            hit_pad.append(np.zeros(hit_dim))
        hit_pad.extend(hit)
        padded_hits.append(np.array(hit_pad))

print(padded_channels.shape)
print(padded_time.shape)
print(padded_labels.shape)
print(len(hits))
print(padded_hits[0].shape)

np.save('./data/' + prefix + 'channels_'+parameter_string + '_' + str(cutoff), padded_channels)
np.save('./data/' + prefix + 'time_' + parameter_string + '_' + str(cutoff), padded_time)
np.save('./data/' + prefix + 'labels_bin_' + parameter_string + "_"+ str(cutoff), padded_labels)
np.save('./data/' + prefix + 'hits_' + parameter_string + '_' + str(cutoff), padded_hits)

print('#Visitors:', len(strings))


