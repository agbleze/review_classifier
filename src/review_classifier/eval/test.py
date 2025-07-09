

#%% TESTING 

# compute the loss & accuracy on the test set using the best available model

classifier.load_state_dict(torch.load(train_state['model_filename']))

classifier = classifier.to(args.device)
dataset.class_weights = dataset.class_weights.to(args.device)
loss_func = nn.CrossEntropyLoss(dataset.class_weights)

dataset.set_split('test')
#######################################        ###################
# batch_generator = generate_batches(dataset, 
#                                    batch_size=args.batch_size, 
#                                    device=args.device)
# running_loss = 0.
# running_acc = 0.
# classifier.eval()

# for batch_index, batch_dict in enumerate(batch_generator):
#     # compute the output
#     y_pred =  classifier(batch_dict['x_data'])
    
#     # compute the loss
#     loss = loss_func(y_pred, batch_dict['y_target'])
#     loss_t = loss.item()
#     running_loss += (loss_t - running_loss) / (batch_index + 1)

#     # compute the accuracy
#     acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
#     running_acc += (acc_t - running_acc) / (batch_index + 1)


model, running_loss, running_acc, mode_bar = compute_model_performance(model=classifier,
                                          mode="test",
                                          dataset=dataset,
                                          loss_func=loss_func,
                                          evaluate_func=compute_accuracy,
                                          mode_bar=None,
                                          #mode_state=train_state
                                          )

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

    
print("Test loss: {};".format(train_state['test_loss']))
print("Test Accuracy: {}".format(train_state['test_acc']))
