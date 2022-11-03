SELECT * FROM "Vehicles_Main";

-- Joining car price, paint_color, and image_url
SELECT "Vehicles_Main".price,
	"Vehicles_Sub".paint_color,
	"Vehicles_Sub".image_url
FROM "Vehicles_Main"
LEFT JOIN "Vehicles_Sub"
ON "Vehicles_Main".id = "Vehicles_Sub".id;