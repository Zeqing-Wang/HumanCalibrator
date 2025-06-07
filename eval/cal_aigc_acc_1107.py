import json
if __name__ == '__main__':
    
    annotation_file = './annotation_aigc_1k.json'
    
    # Ours
    pred_file = './new_inpaint_without_SR_pred.json'

    
    
    
    annotation = json.load(open(annotation_file, 'r'))
    pred = json.load(open(pred_file, 'r'))
    

    
    print(pred_file)
    print(len(pred))
    
    
    
    
    
    acc_dict = {
        'hand':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        },
        'leg':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        },
        'ear':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        },
        'foot':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        },
        'arm':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        },
        'head':{
            'absent_total': 0,
            'detected_absent_total': 0,
            'redundant_total': 0,
            'detected_redundant_total': 0,
            'no_absent_total':0,
            'wrong_detected_absent_total': 0,
            'no_redundant_total': 0,
            'wrong_detected_redundant_total': 0,
        }
    }

            
    for part in ['head', 'ear', 'arm', 'hand', 'leg', 'foot']:
        for key in annotation:
            if key not in pred:
                continue
            if part in annotation[key]['absent']:
                acc_dict[part]['absent_total'] += 1
                if part in pred[key]['absent'] or part+'s' in pred[key]['absent']:
                    acc_dict[part]['detected_absent_total'] += 1
            else:
                acc_dict[part]['no_absent_total'] += 1
                if part in pred[key]['absent'] or part+'s' in pred[key]['absent']:
                    acc_dict[part]['wrong_detected_absent_total'] += 1
                    
            if part in annotation[key]['redundant']:
                acc_dict[part]['redundant_total'] += 1
                if part in pred[key]['redundant'] or part+'s' in pred[key]['redundant']:
                    acc_dict[part]['detected_redundant_total'] += 1
            else:
                acc_dict[part]['no_redundant_total'] += 1
                if part in pred[key]['redundant'] or part+'s' in pred[key]['redundant']:
                    acc_dict[part]['wrong_detected_redundant_total'] += 1
             
    # Calculate detection rate and false detection rate for each category
    for part in acc_dict:
        absent_detection_rate = acc_dict[part]['detected_absent_total'] / acc_dict[part]['absent_total'] if acc_dict[part]['absent_total'] > 0 else 0
        redundant_detection_rate = acc_dict[part]['detected_redundant_total'] / acc_dict[part]['redundant_total'] if acc_dict[part]['redundant_total'] > 0 else 0
        wrong_absent_detection_rate = acc_dict[part]['wrong_detected_absent_total'] / acc_dict[part]['no_absent_total'] if acc_dict[part]['no_absent_total'] > 0 else 0
        wrong_redundant_detection_rate = acc_dict[part]['wrong_detected_redundant_total'] / acc_dict[part]['no_redundant_total'] if acc_dict[part]['no_redundant_total'] > 0 else 0
        
        print(f"Category: {part}")
        print(f"Absent Detection Rate: {absent_detection_rate:.2%}")
        print(f"Redundant Detection Rate: {redundant_detection_rate:.2%}")
        print(f"Wrong Absent Detection Rate: {wrong_absent_detection_rate:.2%}")
        print(f"Wrong Redundant Detection Rate: {wrong_redundant_detection_rate:.2%}")
        print("-" * 40)
    

    
    pass