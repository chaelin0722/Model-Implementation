import numpy as np
arr = np.array(['Welsh_springer_spaniel', 'jigsaw_puzzle', 'rock_beauty', 'spatula', 'goose', 'letter_opener', 'loudspeaker', 'dining_table', 'howler_monkey', 'miniskirt', 'theater_curtain', 'Norfolk_terrier', 'barn_spider', 'radio', 'wallaby', 'junco', 'cup', 'Chihuahua', 'chimpanzee', 'Samoyed', 'quail', 'fireboat', 'castle', 'megalith', 'hatchet', 'cock', 'crate', 'long-horned_beetle', 'rock_crab', 'car_mirror', 'hourglass', 'fur_coat', 'mask', 'tailed_frog', 'moving_van', 'mud_turtle', 'timber_wolf', 'Yorkshire_terrier', 'chain', 'Petri_dish', 'hay', 'boxer', 'ram', 'pencil_sharpener', 'wok', 'lacewing', 'hen', 'lemon', 'apiary', 'barometer', 'moped', 'projectile', 'limousine', 'Staffordshire_bullterrier', 'miniature_pinscher', 'washbasin', 'bulbul', 'throne', 'airliner', 'fire_engine', 'pelican', 'backpack', 'brassiere', 'paddlewheel', 'hippopotamus', 'barracouta', 'web_site', 'little_blue_heron', 'golf_ball', 'dingo', 'night_snake', 'lion', 'volcano', 'altar', 'velvet', 'zucchini', 'banana', 'book_jacket', 'can_opener', 'picket_fence', 'birdhouse', 'gorilla', 'bookcase', 'leatherback_turtle', 'Norwich_terrier', 'garden_spider', 'cowboy_boot', 'fig', 'amphibian', 'banjo', 'French_bulldog', 'sunglass', 'iron', 'toaster', 'Tibetan_terrier', 'digital_clock', 'yawl', 'jaguar', 'mosquito_net', 'sea_lion', 'banded_gecko', 'horse_cart', 'pop_bottle', 'mountain_bike', 'Lhasa', 'croquet_ball', 'hoopskirt', 'green_mamba', 'kelpie', 'cab', 'steam_locomotive', 'typewriter_keyboard', 'bald_eagle', 'leopard', 'patas', 'chest', 'pick', 'coral_reef', 'odometer', 'Angora', 'parking_meter', 'langur', 'analog_clock', 'chickadee', 'Japanese_spaniel', 'missile', 'African_elephant', 'oxygen_mask', 'measuring_cup', 'knot', 'yurt', 'brain_coral', 'neck_brace', 'radio_telescope', 'cannon', 'steel_drum', 'broom', 'mantis', 'coucal', 'ambulance', 'rifle', 'baseball', 'tow_truck', 'pier', 'Bedlington_terrier', 'Scottish_deerhound', 'mongoose', 'liner', 'jacamar', 'folding_chair', 'wild_boar', 'running_shoe', 'garbage_truck', 'chiton', 'guacamole', 'muzzle', 'quilt', 'spindle', 'hotdog', 'jeep', 'starfish', 'mink', 'Doberman', 'pinwheel', 'ptarmigan', 'bullet_train', 'rugby_ball', 'Pembroke', 'ping-pong_ball', 'panpipe', 'teddy', 'sliding_door', 'golden_retriever', 'red_fox', 'ant', 'wallet', 'English_foxhound', 'stethoscope', 'Great_Pyrenees', 'sulphur-crested_cockatoo', 'coffee_mug', 'Brittany_spaniel', 'common_iguana', 'obelisk', 'face_powder', 'barbershop', 'Sealyham_terrier', 'dowitcher', 'scuba_diver', 'malamute', 'agaric', 'gondola', 'collie', 'centipede', 'aircraft_carrier', 'space_shuttle', 'bearskin', 'llama', 'microphone', 'home_theater', 'groenendael', 'monitor', 'screwdriver', 'American_egret', 'saltshaker', 'sweatshirt', 'axolotl', 'otter', 'bakery', 'colobus', 'rain_barrel', 'stretcher', 'titi', 'scoreboard', 'garter_snake', 'cowboy_hat', 'space_heater', 'solar_dish', 'maypole', 'vine_snake', 'common_newt', 'Irish_wolfhound', 'trimaran', 'African_grey', 'passenger_car', 'agama', 'go-kart', 'envelope', 'stove', 'digital_watch', 'Dungeness_crab', 'electric_fan', 'sea_anemone', 'hair_slide', 'tiger_cat', 'beaver', 'pot', 'Saint_Bernard', 'reflex_camera', 'refrigerator', 'shield', 'wreck', 'Cardigan', 'ostrich', 'mobile_home', 'sidewinder', 'American_alligator', 'water_buffalo', 'scale', 'American_lobster', 'great_grey_owl', 'groom', 'Border_terrier', 'rhinoceros_beetle', 'laptop', 'oil_filter', 'guinea_pig', 'crutch', 'indri', 'alligator_lizard', 'palace', 'sunglasses', 'consomme', 'wardrobe', 'ruddy_turnstone', 'box_turtle', 'dishrag', 'disk_brake', 'medicine_chest', 'wood_rabbit', 'Komodo_dragon', 'ocarina', 'punching_bag', 'joystick', 'black-footed_ferret', 'bison', 'ear', 'redshank', 'safe', 'barrel', 'mitten', 'corkscrew', 'puffer', 'wolf_spider', 'Indian_cobra', 'pole', 'alp', 'Saluki', 'trilobite', 'prayer_rug', 'coyote', 'mailbag', 'porcupine', 'harp', 'accordion', 'bath_towel', 'maraca', 'snowmobile', 'ballplayer', 'bassinet', 'dalmatian', 'bassoon', 'Chesapeake_Bay_retriever', 'menu', 'sombrero', 'packet', 'tricycle', 'vizsla', 'fire_screen', 'studio_couch', 'electric_ray', 'terrapin', 'king_snake', 'Mexican_hairless', 'peacock', 'bikini', 'combination_lock', 'school_bus', 'dung_beetle', 'Lakeland_terrier', 'green_lizard', 'hot_pot', 'Rhodesian_ridgeback', 'grille', 'Crock_Pot', 'pool_table', 'valley', 'jinrikisha', 'indigo_bunting', 'cornet', 'parachute', 'spotted_salamander', 'washer', 'king_penguin', 'gibbon', 'candle', 'cauliflower', 'mashed_potato', 'pillow', 'sarong', 'swab', 'cinema', 'screw', 'sock', 'hammer', 'Brabancon_griffon', 'head_cabbage', 'ringneck_snake', 'vending_machine', 'paper_towel', 'tiger', 'remote_control', 'tray', 'police_van', 'bottlecap', 'lynx', 'marmot', 'racer', 'ruffed_grouse', 'hand-held_computer', 'pencil_box', 'soap_dispenser', 'standard_poodle', 'Ibizan_hound', 'toy_poodle', 'ladybug', 'feather_boa', 'hognose_snake', 'custard_apple', 'flute', 'broccoli', 'prairie_chicken', 'Siamese_cat', 'cardoon', 'shovel', 'harvester', 'Blenheim_spaniel', 'cash_machine', 'spider_monkey', 'leaf_beetle', 'trombone', 'Irish_terrier', 'West_Highland_white_terrier', 'triceratops', 'wing', 'kimono', 'chow', 'rotisserie', 'shoji', 'gong', 'forklift', 'catamaran', 'Irish_setter', 'cocker_spaniel', 'German_short-haired_pointer', 'Tibetan_mastiff', 'waffle_iron', 'harvestman', 'black_swan', 'crib', 'vacuum', 'coho', 'acoustic_guitar', 'upright', 'jackfruit', 'safety_pin', 'coffeepot', 'bluetick', 'bolo_tie', 'golfcart', 'canoe', 'ice_cream', 'lens_cap', 'carbonara', 'Granny_Smith', 'bolete', 'red-breasted_merganser', 'hair_spray', 'cassette_player', 'whiskey_jug', 'giant_panda', 'Christmas_stocking', 'cockroach', 'bulletproof_vest', 'spoonbill', 'gyromitra', 'greenhouse', 'gown', 'ringlet', 'harmonica', 'modem', 'projector', 'chocolate_sauce', 'Persian_cat', 'doormat', 'French_horn', 'military_uniform', 'Airedale', 'barn', 'lakeside', 'confectionery', 'sandbar', 'eggnog', 'miniature_schnauzer', 'Windsor_tie', 'milk_can', 'Model_T', 'abaya', 'electric_locomotive', 'dugong', 'lycaenid', 'stinkhorn', 'plane', 'spaghetti_squash', 'loupe', 'grasshopper', 'hand_blower', 'chainlink_fence', 'dogsled', 'trifle', 'bittern', 'dumbbell', 'purse', 'half_track', 'wall_clock', 'keeshond', 'paddle', 'ibex', 'magnetic_compass', 'skunk', 'oscilloscope', 'bathtub', 'manhole_cover', 'lawn_mower', 'jean', 'cabbage_butterfly', 'seat_belt', 'African_crocodile', 'drum', 'Gordon_setter', 'meerkat', 'tractor', 'wig', 'speedboat', 'lotion', 'Madagascar_cat', 'car_wheel', 'sea_urchin', 'shoe_shop', 'drake', 'baboon', 'teapot', 'Shih-Tzu', 'espresso', 'sunscreen', 'mouse', 'hook', 'volleyball', 'rapeseed', 'trailer_truck', 'mountain_tent', 'pug', 'Bouvier_des_Flandres', 'lighter', 'chain_mail', 'Loafer', 'thimble', 'cleaver', 'marmoset', 'assault_rifle', 'African_chameleon', 'Border_collie', 'turnstile', 'minibus', 'Eskimo_dog', 'basset', 'English_setter', 'handkerchief', 'tub', 'grey_fox', 'guenon', 'Pomeranian', 'loggerhead', 'Shetland_sheepdog', 'dough', 'photocopier', 'English_springer', 'streetcar', 'gazelle', 'whistle', 'dragonfly', 'unicycle', 'beaker', 'balance_beam', 'pomegranate', 'basenji', 'boathouse', 'toy_terrier', 'sorrel', 'bannister', 'brambling', 'lifeboat', 'mousetrap', 'Indian_elephant', 'hip', 'dhole', 'French_loaf', 'kit_fox', 'snail', 'radiator', 'rock_python', 'perfume', 'paintbrush', 'toilet_seat', 'black-and-tan_coonhound', 'echidna', 'patio', 'flat-coated_retriever', 'CD_player', 'ladle', 'silky_terrier', 'Greater_Swiss_Mountain_dog', 'tench', 'football_helmet', 'black_grouse', 'white_stork', 'flamingo', 'frilled_lizard', 'poncho', 'stone_wall', 'recreational_vehicle', 'fountain_pen', 'geyser', 'komondor', 'bonnet', 'pajama', 'macaque', 'partridge', 'cricket', 'coil', 'magpie', 'airship', 'container_ship', 'plate', 'wool', 'strawberry', 'badger', 'kuvasz', 'cuirass', 'American_Staffordshire_terrier', 'tennis_ball', 'ox', 'cairn', 'horned_viper', 'Dutch_oven', 'swimming_trunks', 'schipperke', 'Kerry_blue_terrier', 'mortarboard', 'American_chameleon', 'comic_book', 'gas_pump', 'traffic_light', 'Pekinese', 'sundial', 'lumbermill', 'lampshade', 'potpie', 'EntleBucher', 'snorkel', 'bustard', 'crayfish', 'bobsled', 'grand_piano', 'eft', 'vase', 'hartebeest', 'soft-coated_wheaten_terrier', 'Newfoundland', 'thatch', 'power_drill', 'miniature_poodle', 'mosque', 'cliff_dwelling', 'ballpoint', 'ice_lolly', 'bow', 'trolleybus', 'holster', 'Egyptian_cat', 'window_screen', 'pitcher', 'plastic_bag', 'beagle', 'bicycle-built-for-two', 'damselfly', 'rubber_eraser', 'plunger', 'weevil', 'thunder_snake', 'desktop_computer', 'sports_car', 'acorn_squash', 'dome', 'whiptail', 'weasel', 'vulture', 'Great_Dane', 'diamondback', 'macaw', 'bookshop', 'barbell', 'barrow', 'bull_mastiff', 'cloak', 'computer_keyboard', 'sewing_machine', 'artichoke', 'nematode', 'king_crab', 'flagpole', 'curly-coated_retriever', 'great_white_shark', 'barber_chair', 'cougar', 'oboe', 'horizontal_bar', 'space_bar', 'brown_bear', 'gasmask', 'restaurant', 'window_shade', 'plow', 'swing', 'grey_whale', 'stingray', 'carton', 'lorikeet', 'stupa', 'academic_gown', 'German_shepherd', 'hermit_crab', 'scorpion', 'piggy_bank', 'Sussex_spaniel', 'bagel', 'crash_helmet', 'brass', 'iPod', 'abacus', 'espresso_maker', 'standard_schnauzer', 'pay-phone', 'puck', 'orange', 'impala', 'Walker_hound', "yellow_lady's_slipper", 'breastplate', 'hornbill', 'red-backed_sandpiper', 'desk', 'affenpinscher', 'buckeye', 'European_gallinule', 'plate_rack', 'syringe', 'Old_English_sheepdog', 'suspension_bridge', 'spiny_lobster', 'warthog', 'oxcart', 'church', 'rocking_chair', 'hummingbird', 'racket', 'scabbard', 'shower_curtain', 'tiger_beetle', 'tusker', 'lab_coat', 'prison', 'tree_frog', 'three-toed_sloth', 'European_fire_salamander', 'American_black_bear', 'sax', 'sleeping_bag', 'convertible', 'crossword_puzzle', 'toilet_tissue', 'cassette', 'Scotch_terrier', "jack-o'-lantern", 'cheetah', 'sandal', 'pretzel', 'screen', 'drilling_platform', 'butternut_squash', 'siamang', 'tarantula', 'acorn', 'leafhopper', 'slot', 'schooner', 'ski', 'Italian_greyhound', 'tiger_shark', 'black_and_gold_garden_spider', 'caldron', 'seashore', 'daisy', 'maze', 'viaduct', 'carousel', 'notebook', 'mushroom', 'Gila_monster', 'water_ouzel', 'snow_leopard', 'cocktail_shaker', 'whippet', 'limpkin', 'frying_pan', 'motor_scooter', 'pizza', 'bighorn', 'library', 'stole', 'bloodhound', 'water_snake', 'black_stork', 'fountain', 'beer_bottle', 'oystercatcher', 'bathing_cap', 'Leonberg', 'pirate', 'steel_arch_bridge', 'walking_stick', 'pill_bottle', 'lionfish', 'earthstar', 'Siberian_husky', 'chime', 'wine_bottle', 'sea_cucumber', 'Boston_bull', 'wooden_spoon', 'burrito', 'maillot_2', 'chain_saw', 'ashcan', 'slug', 'entertainment_center', 'fiddler_crab', "carpenter's_kit", 'wire-haired_fox_terrier', 'bubble', 'pickup', 'admiral', 'otterhound', 'gar', 'totem_pole', 'Polaroid_camera', 'flatworm', 'cardigan', 'Afghan_hound', 'water_jug', 'tank', 'dam', 'hare', 'green_snake', 'street_sign', 'ski_mask', 'strainer', 'crane', 'robin', 'marimba', 'kite', 'sulphur_butterfly', 'suit', 'promontory', 'butcher_shop', 'crane_bird', 'bucket', 'torch', 'briard', 'anemone_fish', 'cucumber', 'water_tower', 'soup_bowl', 'nail', 'Norwegian_elkhound', 'Australian_terrier', 'clog', 'padlock', 'house_finch', 'jay', 'cicada', 'spider_web', 'switch', 'hyena', 'slide_rule', 'hard_disc', 'cello', 'capuchin', 'cheeseburger', 'beach_wagon', 'cliff', 'dial_telephone', 'bib', 'red_wine', 'corn', 'beacon', 'overskirt', 'necklace', 'vestment', 'breakwater', 'squirrel_monkey', 'goldfinch', 'balloon', 'table_lamp', 'tobacco_shop', 'worm_fence', 'mixing_bowl', 'binder', 'tile_roof', 'borzoi', 'bullfrog', 'goblet', 'Arabian_camel', 'pedestal', 'giant_schnauzer', 'conch', 'armadillo', 'electric_guitar', 'Maltese_dog', 'killer_whale', 'cradle', 'platypus', 'china_cabinet', 'hammerhead', 'stopwatch', 'hen-of-the-woods', 'water_bottle', 'honeycomb', 'soccer_ball', 'jellyfish', 'redbone', 'basketball', 'Band_Aid', 'violin', 'hamster', 'microwave', 'trench_coat', 'Irish_water_spaniel', 'tabby', 'maillot_1', 'boa_constrictor', 'tick', 'mailbox', 'matchstick', 'drumstick', 'sea_snake', 'tripod', 'sturgeon', 'dock', 'ice_bear', 'white_wolf', 'quill', 'thresher', 'triumphal_arch', 'fox_squirrel', 'Dandie_Dinmont', 'orangutan', 'park_bench', 'freight_car', 'zebra', 'planetarium', 'tape_player', 'eel', 'shopping_cart', 'knee_pad', 'bee', 'printer', 'fly', 'umbrella', 'monarch', 'ground_beetle', 'snowplow', "potter's_wheel", 'minivan', 'sloth_bear', 'binoculars', 'wombat', 'albatross', 'mortar', 'lesser_panda', 'African_hunting_dog', 'hog', 'toyshop', 'bow_tie', 'buckle', 'shopping_basket', 'reel', 'bell_pepper', 'beer_glass', 'rule', 'television', 'dishwasher', 'cellular_telephone', 'warplane', 'revolver', 'polecat', 'black_widow', 'guillotine', 'sea_slug', 'papillon', 'chambered_nautilus', 'hamper', 'malinois', 'goldfish', 'stage', 'bell_cote', 'toucan', 'vault', 'nipple', 'grocery_store', 'Appenzeller', 'pickelhaube', 'meat_loaf', 'four-poster', 'file', 'proboscis_monkey', 'Arctic_fox', 'submarine', 'Weimaraner', 'koala', 'spotlight', 'pineapple', 'lipstick', 'American_coot', 'shower_cap', 'Rottweiler', 'organ', 'red_wolf', 'chiffonier', 'bee_eater', 'apron', 'clumber', 'isopod', 'parallel_bars', 'Bernese_mountain_dog', 'monastery', 'Labrador_retriever', 'coral_fungus', 'diaper', 'jersey'])

todict = dict(enumerate(arr,1)) # set 1 for the first

print(todict)