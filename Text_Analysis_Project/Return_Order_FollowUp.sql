SELECT A.buyer_open_uid,A.tid AS 'Return_Order_tid',A.Reference AS 'Return_Reference',A.Category AS 'Return_Category', A.SubCategory AS 'Return_SubCategory', 
A.Collection AS 'Return_Collection',A.created AS'Return_Order_created',B.diff,B.nextOrder_created,B.nextOrder_tid,B.New_Reference,B.New_Category,B.New_SubCategory,B.New_Collection
FROM
	(SELECT tob.tid, p.Reference,p.Category,p.SubCategory,p.Collection,tob.created,tob.buyer_open_uid AS 'buyer_open_uid'
		FROM cartier_bi.t_tmp_crcrefund AS re, cartier_bi.t_order_base AS tob,cartier_bi.pfs_product AS p
		WHERE re.TM = tob.tid
		AND re.Article = p.Reference) AS A
INNER JOIN
    (SELECT T1.*,T2.New_Reference,T2.New_Category,T2.New_SubCategory,T2.New_Collection FROM
		(SELECT ob.buyer_open_uid,ob.created,ob.tid,
			LEAD(ob.created,1) OVER (
				PARTITION BY ob.buyer_open_uid
				ORDER BY ob.created) nextOrder_created,
			LEAD(ob.tid,1) OVER (
				PARTITION BY ob.buyer_open_uid
				ORDER BY ob.created) nextOrder_tid,
			timestampdiff(DAY,ob.created,LEAD(ob.created,1) OVER (
				PARTITION BY ob.buyer_open_uid
				ORDER BY ob.created)) AS 'diff'
			FROM
				cartier_bi.t_order_base as ob) AS T1
		INNER JOIN
			(SELECT p.Reference as 'New_Reference',p.Category AS 'New_Category',p.SubCategory AS 'New_SubCategory',p.Collection AS 'New_Collection',o.order_tid AS 'nextOrder_tid'
				FROM cartier_bi.t_order_item AS o, cartier_bi.pfs_product AS p
				WHERE p.Reference = o.outer_sku_id
			) AS T2
		ON T1.nextOrder_tid = T2.nextOrder_tid
		WHERE T1.diff <= 30) AS B
ON A.tid = B.tid;