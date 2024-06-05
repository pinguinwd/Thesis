# Load the necessary library for plotting
library(ggplot2)
library(plotly)

# Step 1: Read the CSV file into a data frame
data <- read.csv("information.csv", header = TRUE)


# Step 3: Plot each column against the first column
single_char <- function(){
  for(i in 2:ncol(ordered_data)) {
    p <- ggplot(data = ordered_data, aes(x = ordered_data[,1], y = ordered_data[,i])) + 
    geom_line() + 
    labs(title = i,
          x = "Column 1", y = paste("Column", i))
    
    ggsave(
      paste(colnames(ordered_data)[i],'.png'),
      plot = p,
      path = 'D:\\user\\Documents\\unief\\2e master\\Thesis\\Statistical testing\\plots2',
    )
  }
}

multiple_comp1 <- function(){
  # Adjust the ggplot object to include a custom 'text' aesthetic for the hover info
  p <- ggplot(ordered_data, aes(x = Upward_peaks, y = Downward_peaks, 
                                text = paste("<br>Upward_peaks:", Upward_peaks,"<br>Downward_peaks:", Downward_peaks,"<br>Rule:", Rule, "<br>Average:", Average, "<br>Standard Deviation:", Standard_deviation))) +
    geom_jitter(alpha = 0.5, colour = 'blue') +  # Draw the points
    labs(title = "Scatter Plot", x = "Upward Peaks", y = "Downward Peaks") +
    theme_minimal()  # Using a minimal theme for aesthetics
  
  # Convert to ggplotly and specify tooltip info to show on hover
  p_interactive <- ggplotly(p, tooltip = c("text")) %>% 
    layout(hoverlabel = list(bgcolor = "white", 
                             font = list(size = 12)))  # Customize hover label appearance
  
  # Increase the hover area if needed. This is more about adjusting the point sizes to ensure they're easily hoverable.
  # Plotly automatically adjusts hover sensitivity based on point size.
  
  # Save the interactive plot to an HTML file
  htmlwidgets::saveWidget(p_interactive, "D:\\user\\Documents\\unief\\2e master\\Thesis\\Statistical testing\\plots2\\up and downpeaks.html")
  
  # Return or display the interactive plot if needed
  return(p_interactive)
}

multiple_comp2 <- function(){
  # Adjust the ggplot object to include a custom 'text' aesthetic for the hover info
  p <- ggplot(ordered_data, aes(x = Average, y = Standard_deviation, 
                                text = paste("<br>Upward_peaks:", Upward_peaks,"<br>Downward_peaks:", Downward_peaks,"<br>Rule:", Rule, "<br>Average:", Average, "<br>Standard Deviation:", Standard_deviation))) +
    geom_jitter(alpha = 0.5, colour = 'blue') +  # Draw the points
    labs(title = "Scatter Plot", x = "Average", y = "Standard_deviation") +
    theme_minimal()  # Using a minimal theme for aesthetics
  
  # Convert to ggplotly and specify tooltip info to show on hover
  p_interactive <- ggplotly(p, tooltip = c("text")) %>% 
    layout(hoverlabel = list(bgcolor = "white", 
                             font = list(size = 12)))  # Customize hover label appearance
  
  # Increase the hover area if needed. This is more about adjusting the point sizes to ensure they're easily hoverable.
  # Plotly automatically adjusts hover sensitivity based on point size.
  
  # Save the interactive plot to an HTML file
  htmlwidgets::saveWidget(p_interactive, "D:\\user\\Documents\\unief\\2e master\\Thesis\\Statistical testing\\plots2\\Average_VS_Standard_deviation.html")
  
  # Return or display the interactive plot if needed
  return(p_interactive)
}

multiple_comp3 <- function(){
  # Adjust the ggplot object to include a custom 'text' aesthetic for the hover info
  p <- ggplot(ordered_data, aes(x = Average_size_of_peaks, y = Standard_deviation, 
                                text = paste("<br>Upward_peaks:", Upward_peaks,"<br>Downward_peaks:", Downward_peaks,"<br>Rule:", Rule, "<br>Average:", Average, "<br>Standard Deviation:", Standard_deviation))) +
    geom_jitter(alpha = 0.5, colour = 'blue') +  # Draw the points
    labs(title = "Scatter Plot", x = "Average_size_of_peaks", y = "Standard_deviation") +
    theme_minimal()  # Using a minimal theme for aesthetics
  
  # Convert to ggplotly and specify tooltip info to show on hover
  p_interactive <- ggplotly(p, tooltip = c("text")) %>% 
    layout(hoverlabel = list(bgcolor = "white", 
                             font = list(size = 12)))  # Customize hover label appearance
  
  # Increase the hover area if needed. This is more about adjusting the point sizes to ensure they're easily hoverable.
  # Plotly automatically adjusts hover sensitivity based on point size.
  
  # Save the interactive plot to an HTML file
  htmlwidgets::saveWidget(p_interactive, "D:\\PythonProjects\\Thesis\\Statistical analysis\\Statistical testing\\plots2\\Average_peaksize_VS_Standard_deviation.html")
  
  # Return or display the interactive plot if needed
  return(p_interactive)
}

multiple_comp3()

ggplot(data, aes(x = Average, y = Standard.deviation)) +
  geom_point(aes(shape = factor(Wolfram)), color = "grey", alpha = 0.5, size = 5) +  # Use one color, but vary shapes based on 'Wolfram'
  scale_shape_manual(values = c(0, 1, 2, 3), labels = c("Option 1", "Option 2", "Option 3", "Option 4")) +  # Manually specify shapes and labels
  labs(title = "Scatter Plot", x = "Average", y = "Standard Deviation", shape = "Wolfram") +  # Add custom labels, including legend title for shapes
  theme_minimal()   # Using a minimal theme for aesthetics

ggplot(data[data$Wolfram == 4,], aes(x = Average, y = Standard.deviation)) +
  geom_point(color = "black", alpha = 0.5, size = 2) +  # Use one color, but vary shapes based on 'Wolfram'
  labs(x = "Average", y = "Standard Deviation") +  # Add custom labels, including legend title for shapes
  ylim(c(0, 0.1)) +
  xlim(c(0,0.4))# Using a minimal theme for aesthetics
  theme(axis.text=element_text(size=12),
        axis.title=element_text(size=14,face="bold")) +
  theme_minimal()