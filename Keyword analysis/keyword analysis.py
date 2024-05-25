#packages
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy
import networkx as nx
from pyvis.network import Network
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import codecs

#network visualisation

def keywordanalysisbar(file, name):
    #preallocation
    abundance = {}

    #Step 1: read in abstract
    with open(file) as f:
        lines = f.read().replace('\n', '').lower()


    #step 2: define the keywords
    keywords = ['deep learning', 'genetic algorithms', 'fuzzy logic', 'machine learning', ' ai ', 'artificial intelligence']

    #find the keywords in the abstracts
    for keyword in keywords:
        abundance[keyword] = lines.count(keyword.lower())


    keys = list(abundance.keys())
    values = list(abundance.values())

    # Create a bar plot
    plt.bar(keys, values)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.xlabel('Keywords')
    plt.ylabel('Abundance')
    plt.title('Keyword Abundance Bar Plot')

    #adjust subplot parameter
    plt.subplots_adjust(bottom=0.33)

    # Show the plot
    plt.savefig('C:/Users/user/PycharmProjects/Thesis-Wannes/Statistical analysis/keywordfigs/' + name)

def remove_empty_keywords(keyword_string):
    keywords = keyword_string.split(';')
    non_empty_keywords = [keyword.strip() for keyword in keywords if keyword.strip() != '']
    return ';'.join(non_empty_keywords)

def keywordanalysisnetwork(file, sort):
    # Step 1: read in abstract
    with codecs.open(file, encoding='utf-8') as f:
        lines = f.read()

    #regular expression
    if sort == 'authors':
        pattern = re.compile(r'\sAU ([\s\S]*?)AF')

    if sort == 'keywords':
        pattern = re.compile(r'\nDE ([\s\S]*?)[A-Z]{2}')

    matches = pattern.findall(lines)
    


    if sort == 'authors':
        matches2 = []
        for match in matches:
            k = match.replace('\n', ';').replace(';af ',';')
            matches2.append(k)
        matches = matches2

    if sort == 'keywords':
        matches2 = []
        for match in matches:
            k = match.lower()
            matches2.append(k)
        matches = matches2
        

    string_list = [remove_empty_keywords(keywords) for keywords in matches]
    
    forbidden_keywords = ['cellular automata'] #no capital letters

    #Only use the 100 most occuring keywords
    # Step 1: Extract all keywords
    # Step 1: Count occurrences of each keyword
    all_keywords = [keyword.strip() for keywords in string_list for keyword in keywords.split(';')]
    
    #all_keywords = list({keyword.strip(): None for keyword in all_keywords}) #make sure keywords that only differ in a space are not used twice #this code removes doubles
    
    all_keywords = [keyword for keyword in all_keywords if keyword not in forbidden_keywords]
    

    keyword_counts = Counter(all_keywords)

    if sort == 'authors':
        number = 100
    if sort == 'keywords':
        number = 35

    # Step 2: Identify the top 100 most used keywords
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(number)]

    print('top keywords: ', top_keywords)

    # Step 3: Filter each string to include only the top 100 keywords
    filtered_string_list = [
        ';'.join([keyword for keyword in keywords.split(';') if keyword.strip() in top_keywords])
        for keywords in string_list
    ]
    
    # Step 4: Remove elements with zero or one keyword
    filtered_string_list = [keywords for keywords in filtered_string_list if len(keywords.split(';')) >= 1]
    

    if sort == 'keywords':
        #add interesting keywords to the list
        interesting_keywords = []
    if sort == 'authors':
        #add interesting keywords to the list
        interesting_keywords = []

    filtered_string_list += interesting_keywords

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])

     # Iterate through each string in the list
    for string in filtered_string_list:
        keywords = string.split(';')
        keywords = [keyword.strip() for keyword in keywords]  # Remove leading/trailing spaces

        # Generate pairs of connected keywords
        keyword_pairs = [(keywords[i], keywords[j]) for i in range(len(keywords))
                         for j in range(i + 1, len(keywords))]

        # Update the DataFrame with the connections
        for pair in keyword_pairs:
            source, target = pair
            existing_connection = df[(df['Source'] == source) & (df['Target'] == target)].index.values

            if numpy.any(existing_connection):
                # If connection already exists, increment the weight
                df.at[existing_connection[0], 'Weight'] += 1
            else:
                # If connection doesn't exist, add a new row
                df.loc[len(df)] = [source, target, 1, 'undirected']

    # Display the resulting DataFrame
    G = nx.from_pandas_edgelist(df,
                                    source = 'Source',
                                    target = 'Target',
                                    edge_attr= 'Weight')

    # Create a network graph
    net = Network(notebook=True, cdn_resources='in_line')
    net.from_nx(G)

    # Calculate node size and edge color based on the specified attributes
    node_size = df[['Source', 'Target']].apply(lambda x: x.str.strip()).stack().value_counts().to_dict()
    # Normalize 'Weight' values for edge colors
    edge_weights = df['Weight']
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    edge_colors = [cm.viridis(norm(weight)) for weight in edge_weights]


    for node in net.nodes:
        node['size'] = node_size[node['id']]

    # Add edges with edge width attributes
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    edge_widths = [norm(weight) for weight in edge_weights]

    for i in range(len(net.edges)-1):
        net.edges[i]['width'] = edge_widths[i]


    # Generate HTML and save to file
    filename = 'graph_'+sort+'.html'

    html = net.generate_html()
    with open(filename, mode='w', encoding='utf-8') as fp:
        fp.write(html)
    # Display the HTML file in the default web browser
    import webbrowser
    webbrowser.open(filename, new=2)

def netwerkanalysisfuse(filenames, sort):
    with open('outputfile', 'w', encoding='utf-8') as outfile:
        for file in filenames:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.read()
                outfile.write(lines)
    outfile.close()
    keywordanalysisnetwork(outfile.name, sort)

def keywordanalysisnetwork2(file, sort):
    # Step 1: read in abstract
    with codecs.open(file, encoding='utf-8') as f:
        lines = f.read()

    # Define regular expression based on sort type
    pattern = re.compile(r'\nDE ([\s\S]*?)[A-Z]{2}' if sort == 'keywords' else r'\sAU ([\s\S]*?)AF')
    matches = pattern.findall(lines)
    print('len = ', len(matches))

    # Process matches for clean-up and lowercasing if necessary
    if sort == 'authors':
        matches = [match.replace('\n', ';').replace(';af ', ';') for match in matches]
    elif sort == 'keywords':
        matches = [match.lower() for match in matches]
    print('matches: ', matches)

    # Further keyword extraction and filtering
    string_list = [remove_empty_keywords(keywords) for keywords in matches]
    all_keywords = [keyword.strip() for keywords in string_list for keyword in keywords.split(';')]
    all_keywords = [keyword for keyword in all_keywords if keyword not in ['cellular automata']]
    keyword_counts = Counter(all_keywords)

    if sort == 'authors':
        amount_of_keywords = 100
    else:
        amount_of_keywords = 60

    top_keywords = [keyword for keyword, count in keyword_counts.most_common(amount_of_keywords)]

    # Filter strings to include only top keywords and form keyword pairs
    filtered_string_list = [';'.join([keyword for keyword in keywords.split(';') if keyword.strip() in top_keywords]) for keywords in string_list]
    filtered_string_list = [keywords for keywords in filtered_string_list if len(keywords.split(';')) >= 1]

    # Optionally add interesting keywords
    interesting_keywords = [] if sort == 'keywords' else []
    filtered_string_list += interesting_keywords

    # Create DataFrame for network connections
    df = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])
    for string in filtered_string_list:
        keywords = [keyword.strip() for keyword in string.split(';')]
        keyword_pairs = [(keywords[i], keywords[j]) for i in range(len(keywords)) for j in range(i + 1, len(keywords))]
        for source, target in keyword_pairs:
            if (idx := df[(df['Source'] == source) & (df['Target'] == target)].index.values).size > 0:
                df.at[idx[0], 'Weight'] += 1
            else:
                df.loc[len(df)] = [source, target, 1, 'undirected']

    # Visualization using PyVis
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', edge_attr='Weight')
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    # Node and edge adjustments
    # Calculate node sizes based on occurrence in 'Source' and 'Target'
    node_count = df[['Source', 'Target']].stack().value_counts().to_dict()

    # Update node sizes and labels
    scale_factor = 8  # Adjust scale factor for node sizes
    for node in net.nodes:
        node_id = node['id']
        if node_id in node_count:
            node['size'] = node_count[node_id] * scale_factor
            node['title'] = f"{node_id}: {node_count[node_id]} connections"
        else:
            node['size'] = 10  # Default size for nodes that might not have connections listed in node_count
            node['title'] = f"{node_id}: No connections"

    # Adjust edge width and color for visibility
    edge_width_factor = 8  # Adjust to increase edge thickness
    min_weight, max_weight = df['Weight'].min(), df['Weight'].max()
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    for edge in net.edges:
        edge_data = df[(df['Source'] == edge['from']) & (df['Target'] == edge['to'])]
        if not edge_data.empty:
            weight = edge_data['Weight'].iloc[0]
            edge['width'] = norm(weight) * edge_width_factor
            edge_color = cm.viridis(norm(weight))
            edge['color'] = "#{:02x}{:02x}{:02x}".format(*(int(255 * x) for x in edge_color[:3]))

    # Generate HTML and save to file
    filename = 'graph_visualization.html'
    html = net.generate_html()
    with open(filename, 'w', encoding='utf-8') as fp:
        fp.write(html)

    # Display the HTML file in the default web browser
    import webbrowser
    webbrowser.open(filename, new=2)

#keywordanalysisbar('Keyword_analysis_data/67highlycited.txt', '67highlycited')
#keywordanalysisbar('Keyword_analysis_data/1000relevance.txt', '1000relevance')

#keywordanalysisnetwork('Keyword analysis/Keyword_analysis_data/1000_relevant_keywords.txt', 'keywords') #authors or keywords
#keywordanalysisnetwork('Keyword analysis/Keyword_analysis_data/nipt+cell-free.txt', 'authors') #authors or keywords
#keywordanalysisnetwork('Keyword analysis/Keyword_analysis_data/nipt+cell-free.txt', 'keywords')

netwerkanalysisfuse(['Keyword analysis/Keyword_analysis_data/GA.txt','Keyword analysis/Keyword_analysis_data/all.txt'], 'keywords')
#netwerkanalysisfuse(['Keyword_analysis_data/NIPT.txt','Keyword_analysis_data/methylation.txt'], 'authors')

#CA + agent based models


def not_used():
    import requests

    api_key = 'your_api_key'
    base_url = 'https://api.clarivate.com/apis/wos-starter/v1'

    headers = {
        'X-ApiKey': api_key,
        'Content-Type': 'application/json'
    }

    def search_articles(query):
        endpoint = 'search/query'
        params = {
            'databaseId': 'WOS',
            'query': f'TI=({query})',
            'count': 10
        }

        response = requests.get(f'{base_url}{endpoint}', headers=headers, params=params)
        data = response.json()

        return data

    def export_articles(article_ids):
        endpoint = 'data/records'
        params = {
            'databaseId': 'WOS',
            'count': len(article_ids),
            'firstRecord': 1,
            'fields': ['title', 'abstract'],
            'uniqueIds': article_ids
        }

        response = requests.get(f'{base_url}{endpoint}', headers=headers, params=params)
        data = response.json()

        return data

    # Example usage
    query = 'your_search_query'
    search_results = search_articles(query)

    article_ids = [result['uid'] for result in search_results.get('Data', {}).get('Records', [])]
    if article_ids:
        export_data = export_articles(article_ids)
        print(export_data)
    else:
        print("No articles found.")
