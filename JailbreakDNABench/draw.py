import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch # Import Patch for custom legend
from Bio import SeqIO
import collections

# --- Configuration ---
# Define the base directory where the virus datasets are stored
# IMPORTANT: Update this path to the actual location of your virus data folders.
base_dir = "./"

# Map each virus directory name to its broader category
# Ensure directory names here match the actual folder names in base_dir
virus_category_map = {
    "Adenoviridae": "Large DNA viruses",
    "VARV": "Large DNA viruses", # Assuming VARV stands for Variola Virus (Smallpox)
    "Herpesviridae": "Large DNA viruses",
    "HPV": "Large DNA viruses", # Human Papillomavirus
    "B19": "Small DNA viruses", # Parvovirus B19
    "Reovirus": "Double-stranded RNA viruses",
    "SARS-CoV-2": "Positive-strand RNA viruses",
    "MERS-CoV": "Positive-strand RNA viruses",
    "Coronavirus229E": "Positive-strand RNA viruses",
    "CoronavirusHKU1": "Positive-strand RNA viruses",
    "CoronavirusNL63": "Positive-strand RNA viruses",
    "CoronavirusOC43": "Positive-strand RNA viruses",
    "Denguevirus": "Positive-strand RNA viruses",
    "HCV": "Positive-strand RNA viruses", # Hepatitis C Virus
    "Japanese encephalitis virus": "Positive-strand RNA viruses",
    "HIV": "Positive-strand RNA viruses", # Note: HIV is a retrovirus, often grouped with (+)ssRNA
    "Influenza": "Negative-strand RNA viruses",
    "Measles virus": "Negative-strand RNA viruses",
    "Mumpsvirus": "Negative-strand RNA viruses",
    "Rabiesvirus": "Negative-strand RNA viruses",
    "Norovirus": "Enteric RNA viruses", # Often grouped under (+)ssRNA, but keeping separate as requested
    "Poliovirus": "Enteric RNA viruses", # Often grouped under (+)ssRNA
    "Varicella-Zoster Virus": "Large DNA viruses" # Belongs to Herpesviridae
    # Add more mappings if needed
}

# --- Constants ---
CDS_COUNT_CAP = 3 # Maximum CDS count per virus to consider

# --- Data Processing ---

virus_cds_counts_capped = {}
category_cds_counts_capped = collections.defaultdict(int)
# Store details: (category, virus_name, capped_count, original_count)
virus_details = []

print(f"Starting analysis in base directory: {os.path.abspath(base_dir)}")
print(f"Applying CDS count cap: {CDS_COUNT_CAP}")

# Iterate through items in the base directory
for item_name in os.listdir(base_dir):
    item_path = os.path.join(base_dir, item_name)

    # Process only if it's a directory and is in our map
    if os.path.isdir(item_path) and item_name in virus_category_map:
        virus_name = item_name
        category = virus_category_map[virus_name]

        # Determine the path to GenBank files
        patho_path = os.path.join(item_path, "patho")
        gb_file_dir = patho_path if os.path.isdir(patho_path) else item_path

        # Find all .gb files
        try:
            files = [
                os.path.join(gb_file_dir, f)
                for f in os.listdir(gb_file_dir)
                if f.endswith((".gb", ".gbk", ".genbank"))
            ]
        except FileNotFoundError:
            continue # Skip if directory doesn't exist
        except Exception as e:
             print(f"Warning: Error listing files in {gb_file_dir}: {e}")
             continue

        if not files:
            continue # Skip if no GenBank files found

        print(f"Processing {len(files)} file(s) for virus: {virus_name}...")

        original_virus_cds_count = 0
        for file_path in files:
            try:
                for record in SeqIO.parse(file_path, "genbank"):
                    original_virus_cds_count += sum(1 for feature in record.features if feature.type == "CDS")
            except FileNotFoundError:
                 print(f"Error: File not found during parsing: {file_path}")
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")

        # Apply the cap
        capped_cds_count = min(original_virus_cds_count, CDS_COUNT_CAP)

        if capped_cds_count > 0:
            virus_cds_counts_capped[virus_name] = capped_cds_count
            category_cds_counts_capped[category] += capped_cds_count
            virus_details.append({
                'category': category,
                'name': virus_name,
                'capped_count': capped_cds_count,
                'original_count': original_virus_cds_count
            })
        elif original_virus_cds_count > 0:
             print(f"Info: Virus '{virus_name}' had {original_virus_cds_count} CDS, capped count is {capped_cds_count}. Excluding from plot.")
        # else: No CDS found originally

# --- Data Preparation for Plotting ---

# Sort categories by total capped CDS count (descending)
sorted_categories = collections.OrderedDict(
    sorted(category_cds_counts_capped.items(), key=lambda item: item[1], reverse=True)
)

inner_labels = list(sorted_categories.keys())
inner_sizes = list(sorted_categories.values())

if not inner_labels:
    print("Error: No category data found after capping. Cannot generate plot.")
    exit()

# Define base colors for categories (inner ring)
num_categories = len(inner_labels)
colors = plt.cm.tab10(np.linspace(0, 1, min(num_categories, 20)))
if num_categories > 20:
     colors = plt.cm.tab10(np.linspace(0, 1, 20))
     inner_colors = [colors[i % 20] for i in range(num_categories)]
     print("Warning: More than 20 categories, colors will repeat.")
else:
     inner_colors = colors[:num_categories]

category_to_base_color = {label: inner_colors[i] for i, label in enumerate(inner_labels)}

# Sort the individual virus data for the outer ring:
# 1. Primarily by category according to the inner ring's order.
# 2. Secondarily by capped CDS count (descending) within each category.
category_order_map = {category: i for i, category in enumerate(inner_labels)}
sorted_virus_details = sorted(
    virus_details,
    key=lambda item: (category_order_map[item['category']], -item['capped_count'])
)

# Prepare outer ring data and colors with varying alpha
outer_labels_viruses = []
outer_sizes = []
outer_colors_with_alpha = [] # This list will store the final RGBA colors for the outer ring

viruses_by_category = collections.defaultdict(list)
for item in sorted_virus_details:
    viruses_by_category[item['category']].append(item)

min_alpha = 0.3
# We need to build outer_colors_with_alpha in the same order as sorted_virus_details
# Iterate through the already sorted list to generate colors
for item in sorted_virus_details:
    category = item['category']
    items_in_category = viruses_by_category[category] # Get all items for this category
    # Find the index of the current item within its category group (sorted by count desc)
    # This determines its alpha value
    try:
        index_in_category = next(i for i, cat_item in enumerate(items_in_category) if cat_item['name'] == item['name'])
    except StopIteration:
        print(f"Error: Could not find item {item['name']} in its category group for alpha calculation.")
        continue # Skip this item if error occurs

    num_items = len(items_in_category)
    base_color = category_to_base_color[category]
    alpha_values = np.linspace(1.0, min_alpha, num_items) if num_items > 1 else [1.0]
    current_alpha = alpha_values[index_in_category]

    color_with_alpha = to_rgba(base_color, alpha=current_alpha)

    # Append data for the current virus
    outer_labels_viruses.append(item['name'])
    outer_sizes.append(item['capped_count'])
    outer_colors_with_alpha.append(color_with_alpha) # Store the final color with alpha

if not outer_labels_viruses:
    print("Error: No individual virus data with capped CDS counts > 0 found. Cannot generate plot.")
    exit()

# --- Plotting ---

fig, ax = plt.subplots(figsize=(14, 16)) # Keep adjusted height for legend
size = 0.3 # Width of each ring

# Outer circle (individual viruses with varying alpha)
wedges_outer, texts_outer = ax.pie(
    outer_sizes,
    radius=1.0,
    colors=outer_colors_with_alpha, # Use the generated colors with alpha
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=size, edgecolor='w')
)

# Inner circle (main categories)
wedges_inner, texts_inner = ax.pie(
    inner_sizes,
    radius=1.0 - size,
    colors=inner_colors, # Use solid base colors
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=size, edgecolor='w')
)

# --- Legend and Final Touches ---

# Create custom legend handles for individual viruses using their specific outer ring color
legend_handles = []
# Ensure we iterate in the same order as the outer ring segments were created
for i, item in enumerate(sorted_virus_details):
    # Use the exact color (with alpha) from the outer_colors_with_alpha list
    color = outer_colors_with_alpha[i]
    # Create a Patch object with the specific color and the virus name only
    handle = Patch(facecolor=color, edgecolor='grey', # Add a subtle edge for better visibility of patches
                   label=item['name']) # Use only the virus name
    legend_handles.append(handle)


# Add the custom legend
ax.legend(handles=legend_handles,
          title=f"Viruses (Outer Ring)", # Updated title slightly
          loc="center left",
          bbox_to_anchor=(1.05, 0.5),
          fontsize=10,
          title_fontsize=13,
          frameon=False,
          ncol=1) # Adjust ncol if needed


# Optional: Keep or remove the figtext note about transparency
# plt.figtext(0.5, 0.01,
#             f"Outer ring slice transparency decreases with capped CDS count within each category color.",
#             ha="center", fontsize=9, style='italic', wrap=True)


ax.set_title(f'Distribution of CDS Regions (Capped at {CDS_COUNT_CAP}) by Virus Category and Virus', fontsize=16, pad=20)
ax.axis('equal')

plt.tight_layout(rect=[0, 0, 0.80, 0.95]) # Keep adjusted layout for legend

# Save the figure
output_filename = 'virus_dataset_cds_distribution_virus_legend_alpha_color.png' # New filename
try:
    plt.savefig(output_filename, format='png', bbox_inches='tight', dpi=300)
    print(f"\nPlot saved successfully as {output_filename}")
except Exception as e:
    print(f"\nError saving plot: {e}")

# Show the plot
plt.show()

# --- Print Summary Counts ---

print("\n--- Summary Counts (CDS Capped at {}) ---".format(CDS_COUNT_CAP))
print("\nTotal Capped CDS Count per Category:")
for category, count in sorted_categories.items():
    print(f"- {category}: {count}")

print("\nCapped CDS Count per Virus (Plot Order):")
# Print viruses sorted according to the legend/plot order
for item in sorted_virus_details:
     print(f"- {item['name']}: {item['capped_count']} (Original: {item['original_count']})")


print("\nAnalysis complete.")
