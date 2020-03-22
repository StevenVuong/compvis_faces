"""
Mainly to store how to do the prediction; To flesh out if desired
"""


STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
# Must reset test generator before call pred
# Otherwise get outputs in a weird order
test_generator.reset() 
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

# Predicted indices have the predictions
predicted_class_indices=np.argmax(pred,axis=1)

# Map to Original ID's to see what we predicted for what
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

# Save to CSV
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)