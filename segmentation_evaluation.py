
import os
import cv2
import random
import warnings
import argparse
import glob

import numpy as np

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-i", "--submission-dir", type=str, default="polyp_submissions")
parser.add_argument("-o", "--output-dir", type=str, default="results")
parser.add_argument("-t", "--truth-dir", type=str, default="masks")

SUPPORTED_FILETYPES = [".jpg", ".jpeg", ".png"]
CSV_VAL_ORDER = ["accuracy", "jaccard", "dice", "f1", "recall", "precision"]
TEST_FILENAMES = [
    "01d2227c-b9a1-426f-9375-45d787bffd61", "02a47ca5-8c9e-4a25-af7c-0829d3bb6d11", "02c2556d-f11a-4727-9990-5e2544b92788",
    "0a7665a8-2f61-424e-b980-709835faea5e", "0ac12a79-411c-4c97-96bc-7ef235d30197", "0b063da6-d842-41c1-bd33-a6118401429d",
    "0b42905d-356d-499c-9487-6eb166a0bece", "0b51fc05-a393-4d93-a96b-dc339ffc55d6", "0b54479a-f17b-4802-a334-1d016d819441",
    "0b54a431-20c8-4eaf-bd97-9c235b123ffb", "0b5740b8-8a9c-43f5-bf24-083d9ef0e6e1", "0b677b3d-0325-49b3-9a56-83733be084bc",
    "0b777c78-13a1-4afe-a664-26d9fee2ae01", "0b7e3eeb-0461-4c4d-bbe4-be8838d70be5", "0ba7a3db-760f-44b4-9fb4-b624b3e00639",
    "0baf8dcf-965c-4601-9e90-888605e4d08b", "0bb2e271-f2ec-41eb-844f-b9ed824823e3", "0be85999-671e-4057-9a7a-5c6be6977126",
    "0c1a0c1a-a2b1-4403-bae9-27e7cce3ebde", "0c3d86a5-6c13-4f7c-9e26-a5b8159088c2", "0c6cf8be-04b4-4197-af61-be33f5b36ece",
    "0c6f835a-6a59-4a2e-bb3a-5a035ab04d7b", "0c70aef4-83c2-44d7-8e14-5847e3875aab", "0e4f3052-a9d6-4d69-878d-efdbc9f8d804",
    "0e714d8e-395b-47f7-9420-95ebc43f6fb9", "0eb7060d-4ab1-4740-9830-9702441feb0f", "0edc6d1b-f23b-48aa-adea-4610f2f8bceb",
    "0f3293a6-60af-4950-9b48-791bc44eac49", "0f86ccc8-4264-40b0-a235-5dddbf63ff52", "0f907f12-b654-4039-8442-962d9f679598",
    "0fa6ad42-7c5c-4d0f-9e47-7938468b75e4", "0fcebda2-605e-419d-abbd-8224be902a21", "0fd353ad-9416-4f66-b79d-35e3b472e7ee",
    "0fe2bbdc-a46b-4d84-9e03-f0dcc473b500", "1a1b66cb-ba27-4e95-af07-d5e2bde55175", "1a3e796f-0cf4-472d-845f-0ae6c4d50edc",
    "1c236063-7718-45a4-b25c-3aca53a61ca5", "1c286868-03ee-4826-8b56-6948de3bb5ad", "1c3cbc67-aa32-429d-b3c0-68a2ccd65b7e",
    "1c494923-73e5-4e5c-9e49-53c186e6b315", "1c763354-fb73-4d0e-b408-9f16c9bb6000", "1c955c1d-5193-4e6e-88ca-b6e945cc8551",
    "1cc4b51b-e2b9-4192-8788-13dcf5e20fa0", "1cce683c-7756-4154-b4f2-0bed77aa4a29", "1cd16595-28fd-4929-b0aa-b4d0ca3d8361",
    "1d4936b6-6b5b-46ed-b816-efce1e21ad11", "1d7c4516-7edf-4c89-8d12-3ce3c6115d64", "1e0ee7fc-64a0-4254-97b6-629e5b07ba51",
    "1e3712c8-5cf2-423c-8430-6fcc42c12d92", "1e3b0dc6-8e4e-42fc-8f3d-0f2de1023c96", "1e3d18ed-9f65-4abe-911c-0d0e35ffee30",
    "1e8536e0-2da8-4d84-90a3-f6616c6b10ac", "1eac29a3-ea3a-4585-95d5-c63de409513d", "1eb774b8-dfe5-43e7-a1b3-0bef461f9c97",
    "1edbe964-41d0-4262-a798-b9fc7c32f416", "1f39a761-ff59-44a1-b4f9-0b10becf29cf", "1f4f2aa9-8311-4b45-9ac8-4b5e6a02fbdb",
    "2a5f8311-e332-4925-822c-95b2a8184384", "2a7f994b-fb68-4f16-af00-e3a3871a6dce", "2a94f293-689a-4430-93d4-7405eabf3a29",
    "2a9a5f1b-727a-4e6b-b313-c21252c2a908", "2a9bb945-57e2-4fb9-a57c-2ed321e7b5ea", "2a9ef80d-1af8-4f0a-ba11-b00790a79815",
    "2ad8c12f-751e-41c4-bd64-8dea99de4ea0", "2c324452-9bd6-4653-b31c-f32175d706e3", "2c3887a0-88e6-43df-8fb6-61f34ecae93f",
    "2c61fab6-a183-4480-ac99-11f796c6ffe7", "2c6d8dfe-2984-4078-b143-b54a273e43fe", "2c72ec63-aa66-4edd-8dc8-71b1245ce04e",
    "2cb0abd5-0558-4be2-a4ac-7c6da2686099", "2cc92b54-fcbc-4617-b9fc-5477c21b5be9", "2cfe460f-97b2-459a-9b80-95c6444b91fd",
    "2d0a5406-3781-43e9-bed7-14a0f62cc19a", "2d4b255c-f94b-4fcd-ad64-4e62eee102c3", "2d6b7de1-bce7-4353-a86e-5eb4dd880723",
    "2e270c9f-06e0-4900-b910-e65a7e26bddd", "3b0860ef-2e4e-45a2-b1d5-d1aed2a7c131", "3cd5c7cc-721e-47c1-8064-5b3740c33631",
    "3d2ee1c0-4a1a-4567-b2ec-7aa48cddaebb", "3d831d84-6245-41ba-976f-8830fd0c1152", "3d9b8807-c49a-49ba-8ef6-5371dddf8905",
    "3dc35d60-640d-4db9-9593-83a27a619428", "6ff4d484-01f0-466a-b8de-f2adc2f32be8", "7c656e22-4035-42cc-9bda-4c8761f28f73",
    "7d9ecf7d-c848-4b9c-8f60-b41c37ad1175", "8a8e58ff-f403-4d86-bf46-c6a0069a2582", "8abbf663-f590-4455-aad8-87537a0f3928",
    "9a7240e4-48c0-4303-8e22-2b547dc30501", "9a81796c-179d-439b-804d-ce95188e57ba", "e044352a-f821-4ea3-a70a-e704f6055ed9",
    "e0881099-f0b1-4aa4-9503-b139626360d2", "e1610796-08a2-4128-bbef-68a15414482e", "e2181369-a85e-48cd-9945-c635e8d7505c",
    "e23334b6-e02a-41fd-baca-f02a3285be1a", "e2336dc9-dd44-4bc3-8d66-8a05d9c626f8", "e26027d5-bc76-495e-871e-d76e62700490",
    "e2758405-4d17-418c-bac2-4488bc2cef9d", "e28906c9-d311-4f68-85c1-7643ef5ce53f", "e2930eca-5fb1-4baf-b8e6-26f44a46e79c",
    "e3039f59-d9d6-48bf-8f95-38bb722d4054", "e3086272-c857-4466-bf6d-996815cfd4e8", "e3122da2-bc1a-466d-9577-fb87e50b1eb9",
    "e32594e6-7c0b-4953-b113-6932f25ee144", "e3264d6d-f1ba-4e61-aa25-a567965e18a6", "e3386acb-6a7f-43a6-832b-2a69b56b769f",
    "e3929660-68ad-47ca-8a8a-99aa25e8d96f", "e4047456-daab-462f-9794-38b9c8f7d294", "e4240971-477e-4dca-91f1-94beba166160",
    "e51836f5-001a-41c4-82d1-ae1ddd7aec1d", "e5364613-3400-4c15-b755-4f8b63ab7c32", "e54125f9-8aa6-4820-b446-411d2209f075",
    "e55846a2-9927-4d05-a78b-00100e26d55a", "e6014878-e6d4-4d29-aca8-107c1ac63d6a", "e66314ae-8b6e-445d-9f53-c42ae012ce55",
    "e825f9de-9fc4-41a1-9b9a-781c1d16744b", "e831e749-2fa9-4c94-8950-f09861000647", "e853197a-22fb-4cf4-a4e3-f8b7e5a2b522",
    "e8744a7b-12a5-42de-bef4-dbe15c9e8468", "e8762fa3-13d0-46bc-b7ec-2437c7080ff4", "e887718c-0594-4f85-a903-beaa52131edf",
    "e887d53c-0f0a-4be1-a529-63ce173cd7d7", "e8930c78-fbb3-4359-ac10-a2efe733e814", "e901c352-88b6-4ee9-8235-465912e701e8",
    "e911e059-fcdf-4aa5-9a2a-5ea6e20f59e6", "e925f7f8-0699-44dd-9382-e52f8a286a71", "e934d2eb-d491-4fd0-beb8-0634f7b5b057",
    "ea559daa-3995-4fd3-bba9-ac26cf59b098", "ea717261-098d-4473-83b0-c1aea8095c7b", "ea95b1da-a74a-46f9-a56a-e07335c5e2dd",
    "eaba1f8e-b7ce-4171-93fa-9169ed216822", "eabd7e38-c4b6-40fa-b76c-29c39f0adb3c", "eac32756-b853-403c-a067-466e3a67e483",
    "eaca5bf0-d38d-44d5-a4e4-8fac741cb256", "eade39a6-b86a-48f2-9300-2b7d30b64456", "eae03d29-b64d-4d96-a6b5-845e42a305b1",
    "eae81046-a876-4fa1-aa03-9eb7a20d9333", "eaf13300-f5ed-40be-8f92-b3fa2487c65c", "eb2230fb-8b03-4cd1-9630-4c467a629f13",
    "eb8e6e10-f0df-4b14-84c2-5d0756cb1cd8", "eba9440c-e137-4523-8f1c-22934e6a46e2", "ebaa3cf5-9476-4d07-8eb8-1f5561700701",
    "ebbd6667-9b00-4571-821b-0d59beaf9190", "ebc1fdf9-1a3d-4727-9c05-839fe6301bec", "ebce87c9-6617-4ea4-9b9e-475ab77a62ca",
    "ebda56b1-f585-4268-a052-b671e446ceeb", "ebe03e0d-6a32-413d-a3a5-a8bee3e332d0", "ebe31b91-7ed2-476f-a172-d7d422ccd7a2",
    "ebf0124d-ab6e-4d74-a638-d95859a276d7", "ec002b82-5ed0-42ae-a74a-1558e16934b1", "ec2b8c2b-debd-47b7-8e2b-f0eccacbd9c9",
    "ec9c8f63-36d5-4ed2-81d0-6b9794bb2dee", "ecdea6eb-97ce-4ad3-8bc5-175646df285c", "ecfabba0-9198-4ac8-a329-94d9befab7b1",
    "ed4bb384-cbc6-4d17-810a-4e8e2f6b849a", "ed4be1cd-8a00-44eb-8941-eec3b8bb5143", "ed53d9fc-c29d-446e-bb0c-08bce931342a",
    "ed54c3fe-d9ee-49be-9f1f-f31cf1c49f72", "ed5bb98b-e127-431a-89e7-e4426284d384", "ed5c8b65-7160-4e0b-8a40-1f02772bb67b",
    "ed5fb1f6-732d-4a6a-a5a2-c88fee2ef1cd", "ed64c76c-0b9d-498f-a96f-94ea16e4ac95", "ed6dd2e4-b33e-4dcb-b038-75a1f780a220",
    "ed7eab79-9ec6-4dc4-ab32-0468614d3aff", "edf16554-4314-4395-8c57-0ecb1e01ffe5", "edf5470b-db8f-44e5-9ef7-0613c203a468",
    "edff4e72-25eb-4a4e-a393-b96e98f656a8", "ee0f3d45-22d5-41a6-9386-bcd507f4463d", "ee1aa6db-daed-42d8-b3ab-d5d36cc954a0",
    "ee1fbe13-bc69-42c7-9891-37094852949f", "ee2c62dc-9fd1-44c2-9c8b-e855e980942f", "ee2d6d4d-92e6-4e8d-91c5-b6be613e91d8",
    "ee3e06d9-74e4-4c3e-ad17-c653599c1b1a", "ee3fcfa7-4e5d-4659-b781-ac70d7fc1dd3", "ee6cd8ae-3212-4ca8-8ae0-987ec21526ac",
    "ee7d5529-7653-48b5-8ad0-c4432ebb32ea", "eeb57c39-93db-4452-adad-fcea49306c2c", "eec3528f-feaa-457e-aa9d-26f44f722a3e",
    "eec9c113-d467-496c-8987-6e3d826fcb1a", "eeddc984-b6f2-448a-897d-dfc9ff0e4545", "eee4a0bb-24a5-48eb-8161-ace5fde50cb5",
    "eee5c82b-8868-4b35-9070-d4f31dccc7da", "eef691c5-fdfd-478b-9b20-d4630776acba", "ef04370b-f767-4ee5-b109-b2ffdd330e99",
    "ef17501e-22c0-4001-8553-b6cdaf804162", "ef1cca98-e98d-4ab8-8412-a8e9656cb7ef", "ef4fb2e2-fecc-4b46-b874-9ee885023147",
    "ef7acc56-f4ab-4b58-b239-cafe940ca170", "ef8d5827-b524-4815-93bc-e13ff3deac35", "ef8f9c83-e3c0-4275-b6ca-ae55a5958a37",
    "ef907042-2522-4e74-9c57-0c2eca69f902", "ef9ce921-37c3-4ded-8e51-655842cb5826", "efaea1b4-bb14-4bb7-9c6f-faeea24c967c",
    "efb2f003-6f17-47df-964b-7e061eb716d9", "efb54e91-92b2-42fa-9050-b560d00f65c6", "efcebad4-8bf4-4b5b-afd6-0262f22bb5ab",
    "f0cc9fab-5a82-42eb-a9a3-185bcce626ae", "f0df75d5-4c63-4b0c-975a-76b98bc35d08", "f1af53f4-91bb-4d68-9833-f4b9b95e406c",
    "f1df2129-29c6-44b7-80fc-d3626ab0a351", "f3e2f6e6-6288-4d39-9c50-f81e9be13ace"
]

def read_mask(path, target_size=None):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask_shape = mask.shape
    if target_size is not None and mask_shape[0] != target_size[0] and mask_shape[1] != target_size[1]:
        mask = cv2.resize(mask, (target_size[1], target_size[0]))
    if mask.max() > 127:
        mask = mask / 255.0
    mask = mask > 0.5
    mask = mask.astype(np.uint8)
    mask = mask.reshape(-1)
    return mask_shape, mask

def filter_files(path):
    filename, fileext = os.path.splitext(path)
    filename = os.path.basename(filename)

    if not (fileext in SUPPORTED_FILETYPES and filename in TEST_FILENAMES):
        print(fileext in SUPPORTED_FILETYPES, filename in TEST_FILENAMES)
        print("Missing %s" % filename)

    return fileext in SUPPORTED_FILETYPES and filename in TEST_FILENAMES

def dice_score(y_true, y_pred):
    return np.sum(y_pred[y_true == 1] == 1) * 2.0 / (np.sum(y_pred[y_pred == 1] == 1) + np.sum(y_true[y_true == 1] == 1))

def calculate_metrics(y_true, y_pred):
    score_accuracy = accuracy_score(y_true, y_pred)
    score_jaccard = jaccard_score(y_true, y_pred, average="binary")
    score_f1 = f1_score(y_true, y_pred, average="binary")
    score_recall = recall_score(y_true, y_pred, average="binary")
    score_precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    score_dice = dice_score(y_true, y_pred)
    return [score_accuracy, score_jaccard, score_dice, score_f1, score_recall, score_precision]

def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def evaluate_submission(submission_dir, output_dir, ground_truth_dir):

    submission_attributes = os.path.basename(submission_dir).split("_")
    
    team_name = submission_attributes[1]
    run_id = "_".join(submission_attributes[2:-1])
    task_name = submission_attributes[-1]

    team_result_path = os.path.join(output_dir, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)
        
    true_masks = sorted(glob.glob(os.path.join(ground_truth_dir, "*")))
    true_masks = list(filter(filter_files, true_masks))

    pred_masks = sorted(glob.glob(os.path.join(submission_dir, "*")))
    pred_masks = list(filter(filter_files, pred_masks))

    print("Found %i true masks" % len(true_masks))
    print("Found %i pred masks" % len(pred_masks))

    mean_score = []

    detailed_metrics_filename = "%s_%s_%s_detailed_metrics.csv" % (team_name, task_name, run_id)
    average_metrics_filename = "%s_%s_%s_average_metrics.csv" % (team_name, task_name, run_id)

    with open(os.path.join(team_result_path, detailed_metrics_filename), "w") as f:

        f.write("filename;%s\n" % ";".join(CSV_VAL_ORDER))

        assert len(true_masks) == len(pred_masks)

        for index, (y_true_path, y_pred_path) in enumerate(zip(true_masks, pred_masks)):

            print("Progress [%i / %i]" % (index + 1, len(true_masks)), end="\r")
            
            assert get_filename(y_true_path) == get_filename(y_pred_path)

            y_true_shape, y_true = read_mask(y_true_path)
            y_pred_shape, y_pred = read_mask(y_pred_path, y_true_shape)

            metrics = calculate_metrics(y_true, y_pred)

            results_line = "%s;" % get_filename(y_true_path)
            results_line += ";".join(["%0.4f" % score for score in metrics])
            results_line += "\n"
            
            f.write(results_line)

            mean_score.append(metrics)

        print("\n")

    mean_score = np.mean(mean_score, axis=0)

    with open(os.path.join(team_result_path, average_metrics_filename), "w") as f:
        f.write("metric;value\n")
        f.write("\n".join(["%s;%0.4f" % (header, score) for header, score in zip(CSV_VAL_ORDER, mean_score)]))

    with open(os.path.join(output_dir, "%s_all_average_metrics.csv" % task_name), "a") as f:
        f.write("%s;%s;%s;" % (team_name, task_name, run_id))
        f.write(";".join(["%0.4f" % score for score in mean_score]))
        f.write("\n")

if __name__ == "__main__":

    args = parser.parse_args()

    submission_dir = args.submission_dir
    output_dir = args.output_dir
    truth_dir = args.truth_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "%s_all_average_metrics.csv" % os.path.basename(submission_dir)), "w") as f:
        f.write("team-name;task-name;run-id;%s\n" % ";".join(CSV_VAL_ORDER)) 

    for submission_dir in glob.glob(os.path.join(submission_dir, "*")):
        print("Evaluating %s..." % submission_dir)
        evaluate_submission(submission_dir, output_dir, truth_dir)