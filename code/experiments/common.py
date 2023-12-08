import evaluation
import commonUtils


def train_analyze_model(model, data, model_save_path, save_fig_path=None):
    model.fit(data[0], data[2])
    # model = commonUtils.load_pickle_file(model_save_path)
    # predict model
    y_pred = model.predict(data[1])

    # save model
    model.save_model(model_save_path)

    # show confusion matrix for this model
    evaluation.show_confusion_matrix_from_prediction(
        data[3],
        y_pred,
        model.name,
        save_fig_path=save_fig_path,
    )

    return (model.name, data[3], y_pred)
