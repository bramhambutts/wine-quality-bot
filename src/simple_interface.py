from src.classifier import WineClassifier


classifier = WineClassifier()


wine_components = [
    'fixed_acidity',
    'volatile_acidity',
    'citric_acid',
    'residual_sugar',
    'chlorides',
    'free_sulfur_dioxide',
    'total_sulfur_dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
    'color_code'
]

while True:
    print('--- ENTER VALUES TO ESTIMATE WINE QUALITY ---')
    current_wine = {}
    try:
        for component in wine_components:
            component_string = component.replace('_', ' ')
            input_comp = input(f'> Enter the {component_string}: ')
            if input_comp == '':
                current_wine[component] = None
            else:
                current_wine[component] = [float(input_comp)]
    except ValueError:
        print('[FAILURE] Cannot convert string to float type')
        continue
    quality = classifier.classify_wine(current_wine)
    print(f'\n[RESULT] Quality: {quality}\n')