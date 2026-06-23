import pandas as pd
import csv
import io
import re

def pattern_to_text(pattern, names, anchor_label):
    if pd.isna(pattern) or not pattern:
        return ""
    
    # Handle both Law (; ) and Spotify (,) pattern separators
    if "; " in pattern:
        parts = [p.strip() for p in pattern.split(';')]
        id_sep = ";"
    else:
        parts = [p.strip() for p in pattern.split(',')]
        id_sep = ","
        
    if isinstance(names, str):
        # Use detected separator for names
        names_list = [n.strip() for n in names.split(id_sep)]
    else:
        names_list = []
    
    descriptions = []
    for i, part in enumerate(parts):
        name = names_list[i].strip() if i < len(names_list) else "Unknown"
        
        # Spotify Patterns
        if "(Song)-[IN]->(Playlist)-[CREATED_BY]->(User)" in part:
            desc = f"in playlists created by user {name}"
        elif "(Song)-[IN]->(Playlist)-[OF]->(Genre)" in part:
            desc = f"in playlist of genre {name}"
        elif "(Song)-[IN]->(Playlist)" in part:
            desc = f"in playlist {name}"
        elif "(Artist)-[OF]->(Genre)" in part:
            desc = f"of genre {name}"
        elif "(Artist)-[SING]->(Song)-[IN]->(Playlist)-[CREATED_BY]->(User)" in part:
            desc = f"singing songs in playlists created by user {name}"
        elif "(Artist)-[SING]->(Song)-[IN]->(Playlist)-[OF]->(Genre)" in part:
            desc = f"singing songs in playlist of genre {name}"
        elif "(Artist)-[SING]->(Song)-[IN]->(Playlist)" in part:
            desc = f"singing songs in playlist {name}"
        elif "(Artist)-[SING]->(Song)" in part:
            desc = f"singing song {name}"
        elif "(User)-[CREATED_BY]-(Playlist)" in part:
            desc = f"who created playlist {name}"
        
        # Law Patterns (Specific)
        elif "-[CITES]->(Law)-[FROM_DEPARTMENT]->(Department)" in part:
            desc = f"citing a law from department {name}"
        elif "-[CITES]->(Law)-[UNDER_LEGISLATURE]->(Legislature)" in part:
            desc = f"citing a law under legislature {name}"
        elif "-[CITES]->(Law)-[UNDER_GOVERNMENT]->(Government)" in part:
            desc = f"citing a law under government {name}"
        elif "-[HAS_TOPIC]->(Topic)" in part:
            desc = f"related to topic {name}"
        elif "-[FROM_DEPARTMENT]->(Department)" in part:
            desc = f"belonging to department {name}"
        elif "-[UNDER_LEGISLATURE]->(Legislature)" in part:
            desc = f"under legislature {name}"
        elif "-[UNDER_GOVERNMENT]->(Government)" in part:
            desc = f"under government {name}"
        elif "-[HAS_ARTICLE]->(Article)" in part:
            desc = f"containing article {name}"
            
        # Law Patterns (Generic fallback)
        elif "[HAS_ARTICLE]" in part and "[HAS_TOPIC]" in part:
            desc = f"containing articles about {name}"
        elif "[HAS_TOPIC]" in part:
            desc = f"concerning {name}"
        elif "[FROM_DEPARTMENT]" in part:
            desc = f"issued by the {name}"
        elif "[UNDER_LEGISLATURE]" in part:
            desc = f"enacted under {name}"
        elif "[CITES]" in part:
            desc = f"citing {name}"
        
        else:
            clean_part = part.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("->", " ").replace("-", " ")
            desc = f"related to {name} via {clean_part}"
        
        descriptions.append(desc)
    
    return " and ".join(descriptions)

def translate_rule(row):
    anchor = row['Anchor Label']
    body = row['Body']
    body_names = row['Body Node Names']
    head = row['Head']
    head_names = row['Head Node Names']
    support = row.get('Support', 0)
    confidence = row.get('Confidence', 0)
    
    subject = f"{anchor}s"
    body_text = pattern_to_text(body, body_names, anchor)
    head_text = pattern_to_text(head, head_names, anchor)
    head_text = head_text.replace(" and ", " and are ")
    
    return f"{subject} {body_text} that are also {head_text} with support {support} and confidence {confidence}"

def extract_mgr_pattern(pattern_str, names_str, anchor_label):
    """
    Helper function to parse the basic path strings and convert them into 
    MINE GRAPH RULE itemSet syntax with inline properties.
    """
    if pd.isna(pattern_str) or not pattern_str:
        return ""

    # Handle separators
    if "; " in pattern_str:
        parts = [p.strip() for p in pattern_str.split(';')]
        id_sep = ";"
    else:
        parts = [p.strip() for p in pattern_str.split(', ')]
        id_sep = ","

    if isinstance(names_str, str):
        names_list = [n.strip() for n in names_str.split(id_sep)]
    else:
        names_list = []

    itemsets = []
    
    for i, part in enumerate(parts):
        name = names_list[i].strip() if i < len(names_list) else "Unknown"
        
        # Extract nodes and relationships using regex
        nodes = re.findall(r'\\((.*?)\\)', part)
        rels = re.findall(r'\\[(.*?)\\]', part)
        
        if not nodes:
            continue
            
        # Use the full anchor label as the starting point
        mgr_path = f"(:{anchor_label})"
        for j, rel in enumerate(rels):
            # Clean up relation names just in case they have formatting, dropping directional arrows
            rel_clean = rel.replace(':', '').replace('>', '').replace('<', '') 
            target_node = nodes[j+1]
            
            # If this is the terminal node in the path string, tie the specific name to it inline
            if j == len(rels) - 1 and name != "Unknown":
                safe_name = name.replace("'", "\\\\'")
                mgr_path += f"-[:{rel_clean}]-(:{target_node} {{name: '{safe_name}'}})"
            else:
                mgr_path += f"-[:{rel_clean}]-(:{target_node})"
                
        itemsets.append(mgr_path)
        
    return " AND ".join(itemsets)

def translate_to_mgr_syntax(row, rule_name="GeneratedRule"):
    """
    Translates a rule dataframe row into the declarative MINE GRAPH RULE syntax.
    """
    anchor = row.get('Anchor Label', 'UnknownAnchor')
    body = row.get('Body', '')
    body_names = row.get('Body Node Names', '')
    head = row.get('Head', '')
    head_names = row.get('Head Node Names', '')
    support = row.get('Support', 0)
    confidence = row.get('Confidence', 0)
    
    # Parse the strings into GQL-compliant itemSets with inline instance constraints
    body_itemset = extract_mgr_pattern(body, body_names, anchor)
    head_itemset = extract_mgr_pattern(head, head_names, anchor)
    
    # Construct the operator block following the syntax guidelines
    lines = [
        f"MINE GRAPH RULE {rule_name}",
        f"GROUPING ON (:{anchor})"
    ]
    
    if body_itemset:
        lines.append(f"DEFINING BODY AS {body_itemset}")
    
    if head_itemset:
        # Align the head indentation with the body clause if body exists
        prefix = "         HEAD AS " if body_itemset else "DEFINING HEAD AS "
        lines.append(f"{prefix}{head_itemset}")
        
    lines.append(f"EXTRACTING RULES WITH SUPPORT > {support} AND CONFIDENCE > {confidence}")
    
    return "\\n".join(lines)
