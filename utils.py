import matplotlib.pyplot as plt

def plot_kernel(kernel):
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(kernel[0].detach().numpy(), cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap, label='Value')

    # Add labels and title
    plt.title('Heatmap of z2 values')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Add grid lines
    plt.grid(which='major', color='w', linestyle='-', linewidth=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()
