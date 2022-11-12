SELECT * FROM "Vehicles_Main";
SELECT * FROM "Vehicles_Area";

-- Joining car price, paint_color, and image_url
SELECT "Vehicles_Main".price,
	"Vehicles_Sub".paint_color,
	"Vehicles_Sub".image_url
FROM "Vehicles_Main"
LEFT JOIN "Vehicles_Sub"
ON "Vehicles_Main".id = "Vehicles_Sub".id;

-- Setting keys
ALTER TABLE "Vehicles_Area" ADD PRIMARY KEY (state);

ALTER TABLE "Vehicles_Main" ADD FOREIGN KEY (state)
REFERENCES "Vehicles_Area"(state); 

-- Joining id, price, manufacturer, state, and area
SELECT "Vehicles_Main".id,
	"Vehicles_Main".price,
	"Vehicles_Main".manufacturer,
	"Vehicles_Main".state,
	"Vehicles_Area".area
FROM "Vehicles_Main"
LEFT JOIN "Vehicles_Area"
ON "Vehicles_Main".state = "Vehicles_Area".state;