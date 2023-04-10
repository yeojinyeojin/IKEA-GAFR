# VGG Annotator readme

All files are in <root>/annotator folder
1. Open `via.html` in browser
2. Navigate to `Project -> Load` (in the menu bar of the html browser)
3. Select `<root>/annotator/ikea_od_two.json` -> this should automatically load the images and some sample annotations for the first 5 images
4. For annotating, just drag and drop your mouse on the image. The default class is `intricate_sketch` (the sketches that we'll ignore) since this will be more common and for sketched that we care about for 3d reconstruction, change the class to 'plain_sketch' 


Tip: It's easy to accidentally close the tab, so save the project json every 15-30 mins for annotation checkpoints (by going to `Project -> Save`)