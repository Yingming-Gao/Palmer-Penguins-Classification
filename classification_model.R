#install.packages("palmerpenguins")
# install.packages("caret")
library("palmerpenguins")
library(caret)
library(ggplot2)
library(tidyverse)
library(viridis)
# install.packages("paletteer")
library(paletteer)
# Remove sex, year and island, which seem not useful for classification 
cl_data <- penguins %>% 
  select(-sex) %>% 
  select(-year) %>% 
  select(-island) %>%
  drop_na() 
cl_data = cbind( cl_data[,1], scale( cl_data[, 2:5] ))
  
# Split the data set into training set and testing set (80%/20%)
set.seed(333) # Set seed for reproducibility
# Create index for train data
index_train <- createDataPartition(cl_data$species,p=0.8,list = FALSE)
train_data <- cl_data[index_train,] # Train data
test_data <- cl_data[-index_train,] # Test data


# 1. Train KNN(K-nearest neighbor) model, . represents all features
knn_model <- train(species ~ bill_length_mm + bill_depth_mm + 
                     flipper_length_mm + body_mass_g,
                   data = train_data, 
                   method = "knn",tuneLength = 8 )

dev.new(bg = "transparent")
png("plot.png", bg = "transparent")
plot(knn_model) # Plot the training process
dev.off()

print(knn_model)
# Produce predictions, compare with test data
Species <- predict(knn_model, newdata = test_data[2:5])
# Calculate accuracy metrics
knn <- confusionMatrix(Species, test_data$species) 
print(knn)

# Do visualization, pick the first features
# Create a grid of values for prediction
x <- seq(min(cl_data$bill_length_mm), max(cl_data$bill_length_mm), length.out = 200)
y <- seq(min(cl_data$bill_depth_mm), max(cl_data$bill_depth_mm), length.out = 200)

grid <- expand.grid(bill_length_mm = x, bill_depth_mm = y)

grid$flipper_length_mm <- 0; grid$body_mass_g <- 0

# Make predictions on the grid
grid$predicted <- predict(knn_model, newdata = grid)


# Plot the decision boundaries

  ggplot(test_data, aes(x = bill_length_mm, y = bill_depth_mm,color = species)) +
  geom_point(size = 4, shape=17,show.legend = TRUE)+
  geom_point(data = grid, aes(color = predicted), alpha = 0.1, size = 0.5, 
             shape=19, show.legend = FALSE)+
  ggtitle("Decision Boundaries of KNN Model")+
  theme(plot.title = element_text(hjust = 0.5))



#2. Train Nerual Network Model
# install.packages("neuralnet")
library(neuralnet)
# Train the neural network
nn_model <- neuralnet(species ~ ., data = train_data, hidden = c(3, 2))

dev.new(bg = "transparent")
png("plot.png", bg = "transparent")
plot(nn_model)
dev.off()


grid1 <- expand.grid(bill_length_mm = x, bill_depth_mm = y)

grid1$flipper_length_mm <- 0; grid1$body_mass_g <- 0

# Make predictions on the grid
grid1$predicted <- predict(nn_model, newdata = grid1)

# Convert predicted probabilities to class labels
predicted_labels <- max.col(grid1$predicted, "first")  # Obtain the index of the maximum value for each row
grid1$predicted <- factor(predicted_labels, levels = 1:3, 
                           labels = c("Adelie", "Chinstrap", "Gentoo"))  # Convert index to class labels

# Plot the decision boundaries
ggplot(test_data, aes(x = bill_length_mm, y = bill_depth_mm,color = species)) +
  geom_point(size = 4, shape=17,show.legend = TRUE)+
  geom_point(data = grid1, aes(color = predicted), alpha = 0.1, size = 0.5, 
             shape=19, show.legend = FALSE)+
  ggtitle("Decision Boundaries of Neural Network")+
  theme(plot.title = element_text(hjust = 0.5))

# Make predictions on the test dataset
Species <- predict(nn_model, test_data)

# Convert predicted probabilities to class labels
predicted_labels <- max.col(Species, "first")  # Obtain the index of the maximum value for each row
predicted_labels <- factor(predicted_labels, levels = 1:3, 
                           labels = c("Adelie", "Chinstrap", "Gentoo"))  # Convert index to class labels

# Print the predictions
print(predicted_labels)

# Calculate accuracy metrics
nn <- confusionMatrix(predicted_labels, test_data$species)
print(nn)

## Compare the model performances
# install.packages("pROC")
library(pROC)
#install.packages("gplots")
library(gplots)
library(RColorBrewer)
# Create a custom blue color palette
color <- colorRampPalette(c("lightblue", "blue"))(1000)

dev.new(bg = "transparent")
png("plot.png", bg = "transparent")
heatmap.2(knn$table, 
          trace = "none",  # Turn off row/column labels
          col = color,  # Choose color palette
          scale="none",  # Don't scale rows or columns
          key = FALSE,  # Turn off legend
          dendrogram = "none",  # Turn off clustering
          notecol = "white",  # Color of cell annotations
          cellnote = knn$table,  # Cell annotations
          main = "Confusion Matrix Heatmap for KNN Model" # Title
          ,margin=c(10, 10),cexRow = 1.5, # Adjust font size for row labels
          cexCol = 1.5) # Adjust font size for column labels))
dev.off()

dev.new(bg = "transparent")
png("plot.png", bg = "transparent")
heatmap.2(nn$table, 
          trace = "none",  # Turn off row/column labels
          col = color,  # Choose color palette
          scale="none",  # Don't scale rows or columns
          key = FALSE,  # Turn off legend
          dendrogram = "none",  # Turn off clustering
          notecol = "white",  # Color of cell annotations
          cellnote = nn$table,  # Cell annotations
          main = "Confusion Matrix Heatmap for NN Model" # Title
          ,margin=c(10, 10),cexRow = 1.5, # Adjust font size for row labels
          cexCol = 1.5) # Adjust font size for column labels)
dev.off()


