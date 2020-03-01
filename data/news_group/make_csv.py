newsgroups = [fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes')),fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))]

for i in range(len(newsgroups)):
    if i == 0: filename = "test"
    else: filename="train"
    df = pd.DataFrame([newsgroups[i].data, newsgroups[i].target.tolist()]).T
    df.columns = ['text', 'target']
    df["text"] = df["text"].replace("\n|\t"," ", regex=True)
    targets = pd.DataFrame( newsgroups[i].target_names)
    targets.columns=['title']
    out = pd.merge(df, targets, left_on='target', right_index=True)
    out.to_csv('20_newsgroup_'+filename+'.csv')
